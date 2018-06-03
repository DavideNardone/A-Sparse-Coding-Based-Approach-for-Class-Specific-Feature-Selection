from __future__ import division

from sklearn.decomposition import PCA

import numpy as np
import numpy.matlib
np.set_printoptions(threshold=np.inf)
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
import skcuda.misc as misc
import time


class SMBA():

    def __init__(self, data, alpha=10, norm_type=1,
                verbose=False, step=5, thr=[10**-8,-1], max_iter=5000,
                affine=False,
                normalize=True,
                PCA=False, npc=10, GPU=False, device=0):

        self.data = data
        self.alpha = alpha
        self.norm_type=norm_type
        self.verbose = verbose
        self.step = step
        self.thr = thr
        self.max_iter = max_iter
        self.affine = affine
        self.normalize = normalize
        self.device = device
        self.PCA = PCA
        self.npc = npc
        self.GPU = GPU

        self.num_rows = data.shape[0]
        self.num_columns = data.shape[1]

        if(self.GPU==True):
            # self.data = self.data.astype('float32')
            linalg.init()
            # dev = misc.get_current_device()
            # dev = misc.init_device(n=self.device)
            # print misc.get_dev_attrs(dev)


    def computeLambda(self):
        print ('\t\tComputing lambda...')

        T = np.zeros(self.num_columns)

        if (self.GPU == True):

            if not self.affine:

                gpu_data = gpuarray.to_gpu(self.data)
                C_gpu = linalg.dot(gpu_data, gpu_data, transa='T')

                for i in xrange(self.num_columns):
                    T[i] = linalg.norm(C_gpu[i,:])

            else:

                gpu_data = gpuarray.to_gpu(self.data)

                # affine transformation
                y_mean_gpu = misc.mean(gpu_data,axis=1)

                # creating affine matrix to subtract to the data (may encounter problem with strides)
                aff_mat = np.zeros([self.num_rows,self.num_columns]).astype('f')
                for i in xrange(0,self.num_columns):
                    aff_mat[:,i] = y_mean_gpu.get()


                aff_mat_gpu = gpuarray.to_gpu(aff_mat)
                gpu_data_aff = misc.subtract(aff_mat_gpu,gpu_data)

                C_gpu = linalg.dot(gpu_data, gpu_data_aff, transa='T')

                #computing euclidean norm (rows)
                for i in xrange(self.num_columns):
                    T[i] = linalg.norm(C_gpu[i,:])
        else:

            if not self.affine:

                T = np.linalg.norm(np.dot(self.data.T, self.data), axis=1)

            else:
                #affine transformation
                y_mean = np.mean(self.data, axis=1)

                tmp_mat = np.outer(y_mean, np.ones(self.num_columns)) - self.data

                T = np.linalg.norm(np.dot(self.data.T, tmp_mat),axis=1)

        _lambda = np.amax(T)

        return _lambda

    def shrinkL1Lq(self, C1, _lambda):

        D,N = C1.shape
        C2 = []
        if self.norm_type == 1:

            #TODO: incapsulate into one function
            # soft thresholding
            C2 = np.abs(C1) - _lambda
            ind = C2 < 0
            C2[ind] = 0
            C2 = np.multiply(C2, np.sign(C1))
        elif self.norm_type == 2:
            r = np.zeros([D,1])
            for j in xrange(0,D):
                th = np.linalg.norm(C1[j,:]) - _lambda
                r[j] = 0 if th < 0 else th
            C2 = np.multiply(np.matlib.repmat(np.divide(r, (r + _lambda )), 1, N), C1)
        elif self.norm_type == 'inf':
            # TODO: write it
            print ''

        return C2

    def errorCoef(self, Z, C):

        err = np.sum(np.abs(Z-C)) / (np.shape(C)[0] * np.shape(C)[1])

        return err
        # err = sum(sum(abs(Z - C))) / (size(C, 1) * size(C, 2));

    def almLasso_mat_fun(self):

        '''
        This function represents the Augumented Lagrangian Multipliers method for Lasso problem.
        The lagrangian form of the Lasso can be expressed as following:

        MIN{ 1/2||Y-XBHETA||_2^2 + lambda||THETA||_1} s.t B-T=0

        When applied to this problem, the ADMM updates take the form

        BHETA^t+1 = (XtX + rhoI)^-1(Xty + rho^t - mu^t)
        THETA^t+1 = Shrinkage_lambda/rho(BHETA(t+1) + mu(t)/rho)
        mu(t+1) = mu(t) + rho(BHETA(t+1) - BHETA(t+1))

        The algorithm involves a 'ridge regression' update for BHETA, a soft-thresholding (shrinkage) step for THETA and
        then a simple linear update for mu

        NB: Actually, this ADMM version contains several variations such as the using of two penalty parameters instead
        of just one of them (mu1, mu2)
        '''

        print ('\tADMM processing...')

        alpha1 = alpha2 = 0
        if (len(self.reg_params) == 1):
            alpha1 = self.reg_params[0]
            alpha2 = self.reg_params[0]
        elif (len(self.reg_params) == 2):
            alpha1 = self.reg_params[0]
            alpha2 = self.reg_params[1]

        #thresholds parameters for stopping criteria
        if (len(self.thr) == 1):
            thr1 = self.thr[0]
            thr2 = self.thr[0]
        elif (len(self.thr) == 2):
            thr1 = self.thr[0]
            thr2 = self.thr[1]

        # entry condition
        err1 = 10 * thr1
        err2 = 10 * thr2

        start_time = time.time()

        # setting penalty parameters for the ALM
        mu1p = alpha1 * 1/self.computeLambda()
        print("\t\t-Compute Lambda- Time = %s seconds" % (time.time() - start_time))
        mu2p = alpha2 * 1

        mu1 = mu1p
        mu2 = mu2p

        i = 1
        start_time = time.time()
        if self.GPU == True:

            # defining penalty parameters e constraint to minimize, lambda and C matrix respectively
            THETA = misc.zeros((self.num_columns,self.num_columns),dtype='float64')
            lambda2 = misc.zeros((self.num_columns,self.num_columns),dtype='float64')

            gpu_data = gpuarray.to_gpu(self.data)
            P_GPU = linalg.dot(gpu_data,gpu_data,transa='T')

            OP1 = P_GPU
            linalg.scale(np.float32(mu1), OP1)

            OP2 = linalg.eye(self.num_columns)
            linalg.scale(mu2,OP2)


            if self.affine == True:

                print ('\t\tGPU affine...')

                OP3 = misc.ones((self.num_columns, self.num_columns), dtype='float64')
                linalg.scale(mu2, OP3)
                lambda3 = misc.zeros((1, self.num_columns), dtype='float64')

                # TODO: Because of some problem with linalg.inv version of scikit-cuda we fix it using np.linalg.inv of numpy
                A = np.linalg.inv(misc.add(misc.add(OP1.get(), OP2.get()), OP3.get()))

                A_GPU = gpuarray.to_gpu(A)

                while ( (err1 > thr1 or err2 > thr1) and i < self.max_iter):

                    _lambda2 = gpuarray.to_gpu(lambda2)
                    _lambda3 = gpuarray.to_gpu(lambda3)

                    linalg.scale(1/mu2, _lambda2)
                    term_OP2 = gpuarray.to_gpu(_lambda2.get())

                    OP2 = gpuarray.to_gpu(misc.subtract(THETA, term_OP2))
                    linalg.scale(mu2,OP2)

                    OP4 = gpuarray.to_gpu(np.matlib.repmat(_lambda3.get(), self.num_columns, 1))

                    # updating Z
                    BHETA = linalg.dot(A_GPU,misc.add(misc.add(misc.add(OP1,OP2),OP3),OP4))

                    # deallocating unnecessary GPU variables
                    OP2.gpudata.free()
                    OP4.gpudata.free()
                    _lambda2.gpudata.free()
                    _lambda3.gpudata.free()

                    # updating C
                    THETA = misc.add(BHETA,term_OP2)
                    THETA = self.shrinkL1Lq(THETA.get(),1/mu2)
                    THETA = THETA.astype('float64')

                    # updating Lagrange multipliers
                    term_lambda2 = misc.subtract(BHETA, gpuarray.to_gpu(THETA))

                    linalg.scale(mu2,term_lambda2)
                    term_lambda2 = gpuarray.to_gpu(term_lambda2.get())
                    lambda2 = misc.add(lambda2, term_lambda2) # on GPU

                    term_lambda3 = misc.subtract(misc.ones((1, self.num_columns), dtype='float64'), misc.sum(BHETA,axis=0))
                    linalg.scale(mu2,term_lambda3)
                    term_lambda3 = gpuarray.to_gpu(term_lambda3.get())
                    lambda3 = misc.add(lambda3, term_lambda3) # on GPU

                    # deallocating unnecessary GPU variables
                    term_OP2.gpudata.free()
                    term_lambda2.gpudata.free()
                    term_lambda3.gpudata.free()

                    err1 = self.errorCoef(BHETA.get(), THETA)
                    err2 = self.errorCoef(np.sum(BHETA.get(), axis=0), np.ones([1, self.num_columns]))

                    # deallocating unnecessary GPU variables
                    BHETA.gpudata.free()

                    THETA = gpuarray.to_gpu((THETA))

                    # reporting errors
                    if (self.verbose and  (i % self.step == 0)):
                        print('\t\tIteration = %d, ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e' % (i, err1, err2))
                    i += 1

                THETA = THETA.get()

                Err = [err1, err2]
                if(self.verbose):
                    print ('\t\tTerminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e. \n' % (i, err1, err2))

            else:
                print '\t\tGPU not affine'

                # TODO: Because of some problem with linalg.inv version of scikit-cuda we fix it using np.linalg.inv of numpy
                A = np.linalg.inv(misc.add(OP1.get(), OP2.get()))
                A_GPU = gpuarray.to_gpu(A)

                while ( err1 > thr1 and i < self.max_iter):

                    _lambda2 = gpuarray.to_gpu(lambda2)

                    term_OP2 = THETA
                    linalg.scale(mu2, term_OP2)

                    term_OP2 = misc.subtract(term_OP2, _lambda2)

                    OP2 = gpuarray.to_gpu(term_OP2.get())


                    BHETA = linalg.dot(A_GPU, misc.add(OP1 , OP2))

                    linalg.scale(1 / mu2, _lambda2)
                    term_THETA = gpuarray.to_gpu(_lambda2.get())

                    THETA = misc.add(BHETA,term_THETA)
                    THETA = self.shrinkL1Lq(THETA.get(),1/mu2)

                    THETA = THETA.astype('float32')

                    # updating Lagrange multipliers
                    term_lambda2 = misc.subtract(BHETA, gpuarray.to_gpu(THETA))
                    linalg.scale(mu2,term_lambda2)
                    term_lambda2 = gpuarray.to_gpu(term_lambda2.get())
                    lambda2 = misc.add(lambda2, term_lambda2) # on GPU

                    err1 = self.errorCoef(BHETA.get(), THETA)

                    THETA = gpuarray.to_gpu((THETA))

                    # reporting errors
                    if (self.verbose and  (i % self.step == 0)):
                        print('\t\tIteration %5.0f, ||Z - C|| = %2.5e' % (i, err1))
                    i += 1


                THETA = THETA.get()
                Err = [err1, err2]
                if(self.verbose):
                    print ('\t\tTerminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e' % (i, err1))

        else: #CPU version

            # defining penalty parameters e constraint to minimize, lambda and C matrix respectively
            THETA = np.zeros([self.num_columns, self.num_columns])
            lambda2 = np.zeros([self.num_columns, self.num_columns])

            P = self.data.T.dot(self.data)
            OP1 = np.multiply(P, mu1)

            if self.affine == True:

                # INITIALIZATION
                lambda3 = np.zeros(self.num_columns).T

                A = np.linalg.inv(np.multiply(mu1,P) +  np.multiply(mu2, np.eye(self.num_columns, dtype=int)) +  np.multiply(mu2, np.ones([self.num_columns,self.num_columns]) ))

                OP3 = np.multiply(mu2, np.ones([self.num_columns, self.num_columns]))

                while ( (err1 > thr1 or err2 > thr1) and i < self.max_iter):

                    # updating Bheta
                    OP2 = np.multiply(THETA - np.divide(lambda2,mu2), mu2)
                    OP4 = np.matlib.repmat(lambda3, self.num_columns, 1)
                    BHETA = A.dot(OP1 + OP2 + OP3 + OP4 )

                    # updating C
                    THETA = BHETA + np.divide(lambda2,mu2)
                    THETA = self.shrinkL1Lq(THETA, 1/mu2)

                    # updating Lagrange multipliers
                    lambda2 = lambda2 + np.multiply(mu2,BHETA - THETA)
                    lambda3 = lambda3 + np.multiply(mu2, np.ones([1,self.num_columns]) - np.sum(BHETA,axis=0))

                    err1 = self.errorCoef(BHETA, THETA)
                    err2 = self.errorCoef(np.sum(BHETA,axis=0), np.ones([1, self.num_columns]))

                    # mu1 = min(mu1 * (1 + 10 ^ -5), 10 ^ 2 * mu1p);
                    # mu2 = min(mu2 * (1 + 10 ^ -5), 10 ^ 2 * mu2p);

                    # reporting errors
                    if (self.verbose and  (i % self.step == 0)):
                        print('\t\tIteration = %d, ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e' % (i, err1, err2))
                    i += 1

                Err = [err1, err2]

                if(self.verbose):
                    print ('\t\tTerminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e, ||1 - C^T 1|| = %2.5e. \n' % (i, err1,err2))
            else:
                print '\t\tCPU not affine'

                A = np.linalg.inv(OP1 +  np.multiply(mu2, np.eye(self.num_columns, dtype=int)))

                while ( err1 > thr1 and i < self.max_iter):

                    # updating Z
                    OP2 = np.multiply(mu2, THETA) - lambda2
                    BHETA = A.dot(OP1 + OP2)

                    # updating C
                    THETA = BHETA + np.divide(lambda2, mu2)
                    THETA = self.shrinkL1Lq(THETA, 1/mu2)

                    # updating Lagrange multipliers
                    lambda2 = lambda2 + np.multiply(mu2,BHETA - THETA)

                    # computing errors
                    err1 = self.errorCoef(BHETA, THETA)

                    # reporting errors
                    if (self.verbose and  (i % self.step == 0)):
                        print('\t\tIteration %5.0f, ||Z - C|| = %2.5e' % (i, err1))
                    i += 1

                Err = [err1, err2]
                if(self.verbose):
                    print ('\t\tTerminating ADMM at iteration %5.0f, \n ||Z - C|| = %2.5e' % (i, err1))

        print("\t\t-ADMM- Time = %s seconds" % (time.time() - start_time))

        return THETA, Err

    def rmRep(self, sInd, thr):

        '''
        This function takes the data matrix and the indices of the representatives and removes the representatives
        that are too close to each other

        :param sInd: indices of the representatives
        :param thr: threshold for pruning the representatives, typically in [0.9,0.99]
        :return: representatives indices
        '''

        Ys = self.data[:, sInd]

        Ns = Ys.shape[1]
        d = np.zeros([Ns, Ns])

        # Computes a the distance matrix for all selected columns by the algorithm
        for i in xrange(0,Ns-1):
            for j in xrange(i+1,Ns):
                d[i,j] = np.linalg.norm(Ys[:,i] - Ys[:,j])

        d = d + d.T # define symmetric matrix

        dsorti = np.argsort(d,axis=0)[::-1]
        dsort = np.flipud(np.sort(d,axis=0))

        pind = np.arange(0,Ns)
        for i in xrange(0, Ns):
            if np.any(pind==i) == True:
                cum = 0
                t = -1
                while cum <= (thr * np.sum(dsort[:,i])):
                    t += 1
                    cum += dsort[t, i]

                pind = np.setdiff1d(pind, np.setdiff1d( dsorti[t:,i], np.arange(0,i+1), assume_unique=True), assume_unique=True)

        ind = sInd[pind]

        return ind

    def findRep(self,C, thr, norm):

        '''
        This function takes the coefficient matrix with few nonzero rows and computes the indices of the nonzero rows
        :param C: NxN coefficient matrix
        :param thr: threshold for selecting the nonzero rows of C, typically in [0.9,0.99]
        :param norm: value of norm used in the L1/Lq minimization program in {1,2,inf}
        :return: the representatives indices on the basis of the ascending norm of the row of C (larger is the norm of
        a generic row most representative it is)
        '''

        N = C.shape[0]

        r = np.zeros([1,N])

        for i in xrange(0, N):

            r[:,i] = np.linalg.norm(C[i,:],norm)

        nrmInd = np.argsort(r)[0][::-1] #descending order
        nrm = r[0,nrmInd]

        # pick norm indices basing on the thresholding of the 'cumulative norm's sum'
        cssInd = nrmInd[np.cumsum(nrm)/np.sum(nrm) < thr]

        return cssInd, nrmInd


    def admm(self):

        '''
        '''
        # initializing penalty parameters
        self.reg_params = [self.alpha, self.alpha]

        thrS = 0.99
        thrP = 0.95

        #subtract mean from sample
        if self.normalize == True:
            self.data = self.data - np.matlib.repmat(np.mean(self.data, axis=1), self.num_columns,1).T

        self.repInd = []
        if (self.PCA == True):
            print ('\t\tPerforming PCA...')
            pca = PCA(n_components = self.npc)
            self.data = pca.fit_transform(self.data)
            self.num_columns = self.data.shape[0]
            self.num_row = self.data.shape[0]
            self.num_columns = self.data.shape[1]


        self.C,_ = self.almLasso_mat_fun()

        self.sInd, self.nrmInd = self.findRep(self.C, thrS, self.norm_type)

        # custom procedure for removing redundant indices
        # self.repInd = self.rmRep(self.sInd, thrP)
        self.repInd = []


        return self.nrmInd, self.sInd, self.repInd, self.C