from __future__ import division

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

import numpy as np
np.set_printoptions(threshold=np.inf)
import os
import errno
import random as rnd
import itertools
import sys
sys.path.insert(0, './src')

import Loader as lr
import Classifier as clf
import FeatureSelector as fs



def tuning_analysis(fs, n_feats):

    min_var = 99999999
    min_hyp_par = {}

    for curr_fs_name,curr_fs in fs.iteritems():

        voting_matrix = {}
        _res_voting = {}

        combs = curr_fs.keys()
        combs.sort()

        for comb in combs:
            voting_matrix[comb] = np.zeros([1,n_feats])
            value = curr_fs[comb]
            # print ('hyper-params. comb. is %s'%comb)
            curr_var = np.var(value['ACC'])
            if curr_var < min_var:
                min_var = curr_var
                min_hyp_par = comb

        print 'Hyper-params. comb=%s has minimum variance of %s'%(min_hyp_par, min_var)

        combs = curr_fs.keys()
        combs.sort()

        # voting matrix dim: [num_comb, n_feats]
        # voting_matrix = np.zeros([len(combs), n_feats])
        print '\nApplying majority voting...'
        for j in xrange(0,n_feats):
            _competitors = {}
            for comb in combs:
                _competitors[comb] = curr_fs[comb]['ACC'][j]

            #getting the winner accuracy for all the combinations computed
            winners = [comb for m in [max(_competitors.values())] for comb, val in _competitors.iteritems() if val == m]
            for winner in winners:
                voting_matrix[winner][0][j] = 1

        #getting the parameter with largest voting
        for comb in combs:
            _res_voting[comb] = np.sum(voting_matrix[comb][0])

        _max = -9999999
        best_comb = {}
        BS = {}
        for comb in combs:
            if _res_voting[comb] > _max:
                _max = _res_voting[comb]
                best_comb = comb
            print ('Parameters set: '+ comb.__str__() +' got votes: ' + _res_voting[comb].__str__())

        BS[fs_name] = best_comb

        print ('\nBest parameters set found on development set for: ' + fs_name.__str__() + ' is: ' + best_comb.__str__())

    return BS

def create_grid(params):

    comb = []
    for t in itertools.product(*params):
        comb.append(t)

    return comb

def classificationDecisionRule(clf_score, cls, clf_name, target):

    n_classes = len(cls)
    DTS = {}

    for ccn in clf_name:
        hits = []
        res = []
        preds = []

        for i in xrange(0,n_classes):
            # print 'classifier e_' + str(cls[i])

            #ensemble scores on class 'C' for the testing set
            e_th = clf_score['C'+str(cls[i])]['accuracy'][ccn]
            res.append(e_th)

            hits.append((e_th == cls[i]).astype('int').flatten())

        # ensemble scores and hits for the testing set
        ensemble_res = np.vstack(res)
        ensemble_hits = np.vstack(hits)

        # Applying decision rules
        for i in xrange(0, ensemble_hits.shape[1]): # number of sample
            hits = ensemble_hits[:,i]       #it has a 1 in a position whether the classifier e_i has predicted the class w_i for the i-th pattern
            ens_preds = ensemble_res[:,i]   #it's simply the predictions of all the trained classifier for the i-th pattern
            cond = np.sum(hits)             #count the number of true positive for the i-th pattern

            if cond == 1: #rule 1
                pred = cls[np.where(hits==1)[0].squeeze()] #retrieve the cls for the 'only' true positive
                preds.append(pred)

            elif cond == 0 or cond > 1:  # rule 1-2 (tie)

                # we find the majority votes (frequency) among all classifier (e.g., ) [[4 2][5 1][6 2][7 2]]
                unique, counts = np.unique(ens_preds, return_counts=True)
                maj_rule = np.asarray((unique, counts)).T

                # we find the 'majority' index, then its class
                ind_max = np.argmax(maj_rule[:, 1])
                pred = maj_rule[ind_max, 0]
                max = maj_rule[ind_max, 1]

                # we look for a 'tie of the tie', then we look for the majority class among all the tied classes
                tied_cls =  np.where(maj_rule[:, 1] == max)[0]
                if ( len(np.where(maj_rule[:, 1] == max)[0]) ) > 1: #tie of the tie
                    pred = maj_rule[tied_cls,0]

                    # pick one tied cls randomly
                    pred = pred[rnd.randint(0,len(pred)-1)]
                    preds.append(pred)

                else:
                    preds.append(pred)

        #compute accuracy
        test_score = accuracy_score(target, preds)

        dic_test_score = {
            ccn: test_score
        }

        DTS.update(dic_test_score)

    return DTS

def checkFolder(root, path_output):

    #folders to generate recursively
    path = root+'/'+path_output

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



if __name__ == '__main__':

    ''' LOADING ANY DATASET '''
    dataset_dir = '/dataset'
    dataset_type = '/BIOLOGICAL'
    dataset_name = '/LUNG_DISCRETE'

    resample = True

    path_data_folder = dataset_dir + dataset_type + dataset_name
    path_data_file = path_data_folder + dataset_name

    variables = ['X', 'Y']
    # NB: If you get an error such as: 'Please use HDF reader for matlab v7.3 files',please change the 'format variable' to 'matlab_v73'
    D = lr.Loader(file_path=path_data_file,
                  format='matlab',
                  variables=variables,
                  name=dataset_name[1:]
                  ).getVariables(variables=variables)

    dataset = lr.Dataset(D['X'], D['Y'])

    # check if the data are already standardized, if not standardize it
    dataset.standardizeDataset()

    n_classes = dataset.classes.shape[0]
    cls = np.unique(dataset.classes)

    num_min_cls = 9999999
    if resample == True:

        print ('Dataset before resampling %s w/ size: %s and number of classes: %s---> %s' % (
        dataset_name[1:], dataset.data.shape, n_classes, cls))

        # discriminating classes the whole dataset
        dataset_train = lr.Dataset(dataset.data, dataset.target)
        dataset_train.separateSampleClass()
        data, target = dataset_train.getSampleClass()

        for i in xrange(0, n_classes):
            print ('# sample for class C' + str(i + 1) + ':', data[i].shape)
            if data[i].shape[0] < num_min_cls:
                num_min_cls = data[i].shape[0]

        resample = '/BALANCED'
        print 'Re-sampling dataset...'
        dataset.data, dataset.target = SMOTE(kind='regular', k_neighbors=num_min_cls-1).fit_sample(dataset.data, dataset.target)
    else:
        resample = '/UNBALANCED'

    # shuffling data
    dataset.shufflingDataset()

    n_classes = dataset.classes.shape[0]
    cls = np.unique(dataset.classes)

    print ('Dataset %s w/ size: %s and number of classes: %s---> %s' %(dataset_name[1:], dataset.data.shape, n_classes, cls))

    # discriminating classes the whole dataset
    dataset_train = lr.Dataset(dataset.data, dataset.target)
    dataset_train.separateSampleClass()
    data, target = dataset_train.getSampleClass()

    for i in xrange(0, n_classes):
        print ('# sample for class C' + str(i + 1) + ':', data[i].shape)


################################### TUNING PARAMS ###################################

    FS = {}

    #CLASSIFIERS
    clf_name = [
        "SVM"
        # "Decision Tree",
        # "KNN"
    ]
    model = [
        SVC(kernel="linear")
        # DecisionTreeClassifier(max_depth=5),
        # KNeighborsClassifier(n_neighbors=1),
    ]

    max_num_feat = 300
    step = 1

    # initializing feature selector parameters
    params = {
        # the smaller alpha the sparser C matrix (fewer representatives)
        'SMBA':
            {
                'alpha':        5, #typically alpha in [2,50]
                'norm_type':    1,
                'max_iter':     3000,
                'thr':          [10**-8],
                'type_indices': 'nrmInd',
                'normalize':    False,
                'GPU':          False,
                'device':       0,
                'PCA':          False,
                'verbose':      False,
                'step':         1,
                'affine':       False,
            },
        'RFS':
            {
                'gamma': 0
            },
        'll_l21':
            {
                'z': 0
            },
        'ls_l21':
            {
                'z': 0
            },
        'Relief':
            {
                'k': 0
            },
        'MRMR':
            {

            'num_feats': max_num_feat
            },
        'MI':
            {
            'n_neighbors': 0
            },
        # the bigger is alpha the sparser is the C matrix (fewer representatives)
        'EN':
            {
                'alpha': 1,  # default value is 1
            },
        # the bigger is alpha the sparser is the C matrix (fewer representatives)
        'LASSO':
            {
                'alpha': 1  # default value is 1
            }
    }


    slb_fs = {

        'LASSO': fs.FeatureSelector(name='LASSO', tp='SLB', params=params['LASSO']),
        'EN': fs.FeatureSelector(name='EN', tp='SLB', params=params['EN']),
        'SMBA':     fs.FeatureSelector(name='SMBA', tp='SLB', params=params['SMBA']),
        'RFS':    fs.FeatureSelector(name='RFS', tp='SLB',params=params['RFS']),
        'll_l21': fs.FeatureSelector(name='ll_l21', tp='SLB',params=params['ll_l21']), #injection not working
        'ls_l21': fs.FeatureSelector(name='ls_l21', tp='SLB',params=params['ls_l21']),

        'Relief': fs.FeatureSelector(name='Relief', tp='filter', params=params['Relief']),
        'MRMR':     fs.FeatureSelector(name='MRMR', tp='ITB', params=params['MRMR']),
        'MI': fs.FeatureSelector(name='MI', tp='filter', params=params['MI'])
    }

    tuned_parameters = {

        'LASSO':    {'alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]},
        'EN':       {'alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]},
        'SMBA':     {'alpha': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
        'RFS':      {'gamma': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]},
        'll_l21':   {'z': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]},
        'ls_l21':   {'z': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]},
        'Relief':   {'k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        'MRMR':     {'num_feats': [max_num_feat]},
        'MI':       {'n_neighbors': [1, 2, 3, 5, 7, 10]}
    }


    if max_num_feat > dataset.data.shape[1]:
        max_num_feat = dataset.data.shape[1]

    print ('\nMax number of features to use: ', max_num_feat)

    #setting the parameters for k-fold CV
    k_fold = 5

    X = dataset.data
    y = dataset.target
    kf = KFold(n_splits=k_fold)

tuning_type = 'CSFS-PAPER'

################################### TFS  ###################################

if tuning_type == 'TFS':

    res_path_output = '/TFS/RESULTS/'

    # tuning process on all feature selector
    for fs_name, fs_model in slb_fs.iteritems():

        # check whether the 'curr_res_fs_fold' directory exists, otherwise create it
        # curr_res_output_fold = path_data_folder + '/' + res_path_output + '/TUNING/' + resample + '/'+ fs_name
        # checkFolder(path_data_folder, res_path_output + '/TUNING/' + resample + '/'+ fs_name)

        print '\nTuning hyper-parameters on ' +fs_name.__str__()+ ' for accuracy by means of k-fold CV...'
        FS.update({fs_name: {}})
        comb = []
        params_name = []

        for name, tun_par in tuned_parameters[fs_name].iteritems():
            comb.append(tun_par)
            params_name.append(name)

        #create all combinations parameters
        combs = create_grid(comb)

        n_iter = 1
        #loop on each combination
        for comb in combs:

            FS[fs_name].update({comb: {}})
            CV = np.ones([k_fold, max_num_feat])*0
            avg_scores = []
            std_scores = []

            print ('\tComputing '+n_iter.__str__() +'-th combination...')

            # set i-th parameters combination parameters for the current feature selector
            fs_model.setParams(comb,params_name,params[fs_name])

            cc_fold = 0
            for train_index, test_index in kf.split(X):

                kth_scores = []

                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                idx = fs_model.fit(X_train, y_train)
                # print idx
                # idx = list(range(1, 20))

                #classification step on the first max_num_feat
                for n_rep in xrange(step, max_num_feat + step, step):  # first n_rep indices

                    X_train_fs = X_train[:, idx[0:n_rep]]
                    X_test_fs = X_test[:, idx[0:n_rep]]

                    _clf = clf.Classifier(names=clf_name, classifiers=model)
                    DTS = _clf.train_and_classify(X_train_fs, y_train, X_test_fs, y_test)

                    _score = DTS['SVM']
                    kth_scores.append(_score) # it contains the max_num_feat scores for the k-th CV fold

                CV[cc_fold,:] = kth_scores
                cc_fold += 1

            avg_scores = np.mean(CV, axis=0)
            std_scores = np.std(CV, axis=0)

            FS[fs_name][comb]['ACC'] = avg_scores
            FS[fs_name][comb]['STD'] = std_scores

            n_iter +=1

    #tuning analysis
    print 'Applying tuning analysis...'
    num_feat = 10
    best_params = tuning_analysis(FS,num_feat)

    print 'Saving results...\n'
    # curr_res_output_fold = path_data_folder + '/' + res_path_output + '/TUNING/' + resample
    # checkFolder(path_data_folder, res_path_output + '/TUNING/' + resample)
    #
    # with open(curr_res_output_fold + '/' + 'best_params.pickle', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif tuning_type == 'CSFS':

    res_path_output = '/CSFS/RESULTS/'

    # tuning process on all feature selector
    for fs_name, fs_model in slb_fs.iteritems():

        # check whether the 'curr_res_fs_fold' directory exists, otherwise create it
        # curr_res_output_fold = path_data_folder + '/' + res_path_output + '/TUNING/' + resample + '/'+ fs_name
        # checkFolder(path_data_folder, res_path_output + '/TUNING/' + resample + '/'+ fs_name)

        print '\nTuning hyper-parameters on ' +fs_name.__str__()+ ' for accuracy by means of k-fold CV...'
        FS.update({fs_name: {}})
        comb = []
        params_name = []

        for name, tun_par in tuned_parameters[fs_name].iteritems():
            comb.append(tun_par)
            params_name.append(name)

        #create all combinations parameters
        combs = create_grid(comb)

        n_iter = 1
        #loop on each combination
        for comb in combs:

            FS[fs_name].update({comb: {}})
            CV = np.ones([k_fold, max_num_feat])*0
            avg_scores = []
            std_scores = []

            print ('\tComputing '+n_iter.__str__() +'-th combination...')

            # set i-th parameters combination parameters for the current feature selector
            fs_model.setParams(comb,params_name,params[fs_name])

            cc_fold = 0
            # k-fold CV
            for train_index, test_index in kf.split(X):

                kth_scores = []

                csfs_res = {}
                cls_res = {}
                k_computing_time = 0

                for i in xrange(0, n_classes):
                    cls_res = {

                        'C' + str(cls[i]): {}
                    }
                    csfs_res.update(cls_res)

                X_train_kth, X_test_kth = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                ''' For the training data in each class we find the representative features and use them as a best subset feature
                    (in representing each class sample) to perform classification
                '''

                # check whether the 'curr_res_fs_fold' directory exists, otherwise create it
                # curr_res_fs_fold = path_data_folder + '/' + fs_path_output + '/' + fs_name + resample
                # checkFolder(path_data_folder, fs_path_output + '/' + fs_name + resample)

                # discriminating classes for the k-th fold of the training set
                data_train = lr.Dataset(X_train_kth, y_train)
                data_train.separateSampleClass()
                ktrain_data, ktrain_target = data_train.getSampleClass()

                for i in xrange(0, n_classes):

                    idx = fs_model.fit(ktrain_data[i], ktrain_target[i])

                    csfs_res['C' + str(cls[i])]['idx'] = idx
                    csfs_res['C' + str(cls[i])]['params'] = params[fs_name]

                # learning a classifier (ccn) for each subset of 'n_rep' feature
                for j in xrange(0, max_num_feat):
                    n_rep = j + 1  # first n_rep indices

                    for i in xrange(0, n_classes):
                        # get subset of feature from the i-th class
                        idx = csfs_res['C' + str(cls[i])]['idx']

                        X_train_fs = X_train_kth[:, idx[0:n_rep]]

                        _clf = clf.Classifier(names=clf_name, classifiers=model)
                        _clf.train(X_train_fs, y_train)

                        csfs_res['C' + str(cls[i])]['accuracy'] = _clf.classify(X_test_kth[:, idx[0:n_rep]], y_test)

                    DTS = classificationDecisionRule(csfs_res, cls, clf_name, y_test)

                    _score = DTS['SVM']
                    kth_scores.append(_score)
                # print kth_scores

                CV[cc_fold,:] = kth_scores
                cc_fold += 1

            avg_scores = np.mean(CV, axis=0)
            std_scores = np.std(CV, axis=0)

            # print avg_scores

            FS[fs_name][comb]['ACC'] = avg_scores
            FS[fs_name][comb]['STD'] = std_scores

            n_iter +=1

    #tuning analysis
    print 'Applying tuning analysis...'
    num_feat = 10
    best_params = tuning_analysis(FS,num_feat)

    print best_params

    # print 'Saving results...\n'
    # curr_res_output_fold = path_data_folder + '/' + res_path_output + '/TUNING/' + resample
    # checkFolder(path_data_folder, res_path_output + '/TUNING/' + resample)
    #
    # with open(curr_res_output_fold + '/' + 'best_params.pickle', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif tuning_type == 'CSFS-PAPER':

    res_path_output = '/CSFS_PAPER/RESULTS/'

    # tuning process on all feature selector
    for fs_name, fs_model in slb_fs.iteritems():

        # check whether the 'curr_res_fs_fold' directory exists, otherwise create it
        # curr_res_output_fold = path_data_folder + '/' + res_path_output + '/TUNING/' + resample + '/'+ fs_name
        # checkFolder(path_data_folder, res_path_output + '/TUNING/' + resample + '/'+ fs_name)

        print '\nTuning hyper-parameters on ' +fs_name.__str__()+ ' for accuracy by means of k-fold CV...'
        FS.update({fs_name: {}})
        comb = []
        params_name = []

        for name, tun_par in tuned_parameters[fs_name].iteritems():
            comb.append(tun_par)
            params_name.append(name)

        #create all combinations parameters
        combs = create_grid(comb)

        n_iter = 1
        #loop on each combination
        for comb in combs:

            FS[fs_name].update({comb: {}})
            CV = np.ones([k_fold, max_num_feat])*0
            avg_scores = []
            std_scores = []

            print ('\tComputing '+n_iter.__str__() +'-th combination...')

            # set i-th parameters combination parameters for the current feature selector
            fs_model.setParams(comb,params_name,params[fs_name])

            cc_fold = 0
            # k-fold CV
            for train_index, test_index in kf.split(X):

                kth_scores = []

                csfs_res = {}
                cls_res = {}
                k_computing_time = 0

                for i in xrange(0, n_classes):
                    cls_res = {

                        'C' + str(cls[i]): {}
                    }
                    csfs_res.update(cls_res)

                X_train_kth, X_test_kth = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # CLASS BINARIZATION
                lb = label_binarize(cls, classes=y_train)

                for i in xrange(0, n_classes):
                    num_min_cls = 9999999
                    k_neighbors = 5

                    # discriminating classes the whole dataset
                    dataset_train = lr.Dataset(X_train_kth, lb[i])
                    dataset_train.separateSampleClass()
                    data, target = dataset_train.getSampleClass()

                    for j in xrange(0, 2):
                        if data[j].shape[0] < num_min_cls:
                            num_min_cls = data[j].shape[0]

                    if num_min_cls == 1:
                        num_min_cls += 1

                    # CLASS BALANCING
                    data_cls, target_cls = SMOTE(kind='regular',k_neighbors=num_min_cls-1).fit_sample(X_train_kth, lb[i])

                    # Performing feature selection on each class

                    idx = fs_model.fit(data_cls, target_cls)

                    csfs_res['C' + str(cls[i])]['idx'] = idx
                    csfs_res['C' + str(cls[i])]['params'] = params[fs_name]

                # Classification
                ens_class = {}
                # learning a classifier (ccn) for each subset of 'n_rep' feature
                for n_rep in xrange(step, max_num_feat + step, step):  # first n_rep indices

                    for i in xrange(0, n_classes):
                        # get subset of feature from the i-th class
                        idx = csfs_res['C' + str(cls[i])]['idx']

                        X_train_fs = X_train_kth[:, idx[0:n_rep]]

                        _clf = clf.Classifier(names=clf_name, classifiers=model)
                        _clf.train(X_train_fs, y_train)

                        csfs_res['C' + str(cls[i])]['accuracy'] = _clf.classify(X_test_kth[:, idx[0:n_rep]], y_test)

                    DTS = classificationDecisionRule(csfs_res, cls, clf_name, y_test)

                    _score = DTS['SVM']
                    kth_scores.append(_score)

                CV[cc_fold, :] = kth_scores
                cc_fold += 1

            avg_scores = np.mean(CV, axis=0)
            std_scores = np.std(CV, axis=0)

            FS[fs_name][comb]['ACC'] = avg_scores
            FS[fs_name][comb]['STD'] = std_scores

            n_iter += 1

    #tuning analysis
    print 'Applying tuning analysis...'
    num_feat = 10
    best_params = tuning_analysis(FS,num_feat)

    print best_params

    # print 'Saving results...\n'
    # curr_res_output_fold = path_data_folder + '/' + res_path_output + '/TUNING/' + resample
    # checkFolder(path_data_folder, res_path_output + '/TUNING/' + resample)
    #
    # with open(curr_res_output_fold + '/' + 'best_params.pickle', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    print 'Wrong tuning type selected'
