from __future__ import division

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE #dependency

import numpy as np
np.set_printoptions(threshold=np.inf)
import random as rnd
import time
import os
import errno
import pickle
import sys

# sys.path.insert(0, './src')
import Loader as lr
import Dataset as ds
import Classifier as i_clf
import FeatureSelector as fs


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

def OCC_DecisioneRule(clf_score, cls, clf_name, target):

    n_classes = len(cls)

    DTS = {}
    for ccn in clf_name:
        hits = []
        res = []
        preds = []

        for i in xrange(0,n_classes):
            e_th =  np.asarray(clf_score['C'+str(cls[i])]['accuracy'][ccn])

            e_th[np.where(e_th==-1)] = 0

            hits.append(e_th)

        ensemble_hits = np.vstack(hits)

        for i in xrange(0, ensemble_hits.shape[1]): # number of sample
            hits = ensemble_hits[:,i]
            cond = np.sum(hits)

            if cond == 1: #rule 1
                pred = np.where(hits==1)[0]
                pred = cls[pred][0]
                preds.append(pred)
            elif cond == 0: #rule 2 (tie among all OCC)
                pred = cls[rnd.randint(0, len(cls) - 1)]
                preds.append(pred)
            elif cond > 0:
                tied_cls = np.where(hits==1)[0]
                pred = tied_cls[rnd.randint(0, len(tied_cls) - 1)]
                preds.append(pred)

        test_score = accuracy_score(target, preds)

        dic_test_score = {
            ccn: test_score
        }

        DTS.update(dic_test_score)

    return DTS

def classificationDecisionRule(clf_score, cls, clf_name, target):

    n_classes = len(cls)
    DTS = {}

    for ccn in clf_name:
        hits = []
        res = []
        preds = []

        for i in xrange(0,n_classes):

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

def main():
    ''' LOADING ANY DATASET '''
    dataset_dir = '/dataset'
    dataset_type = '/BIOLOGICAL'
    dataset_name = '/WISCONSIN'

    #this variable decide whether to balance or not the dataset
    resample = True
    p_step = 1

    # defining directory paths for saving partial and complete result
    path_data_folder = dataset_dir + dataset_type + dataset_name
    path_data_file = path_data_folder + dataset_name
    variables = ['X', 'Y']

    print ('%d.Loading and pre-processing the data...\n' % p_step)
    p_step += 1
    # NB: If you get an error such as: 'Please use HDF reader for matlab v7.3 files',please change the 'format variable' to 'matlab_v73'
    D = lr.Loader(file_path=path_data_file,
                  format='matlab',
                  variables=variables,
                  name=dataset_name[1:]
                  ).getVariables(variables=variables)

    dataset = ds.Dataset(D['X'], D['Y'])

    n_classes = dataset.classes.shape[0]
    cls = np.unique(dataset.classes)

    # check if the data are already standardized, if not standardize it
    dataset.standardizeDataset()

    # re-sampling dataset
    num_min_cls = 9999999
    print ('%d.Class-sample separation...\n' % p_step)
    p_step += 1
    if resample == True:

        print ('\tDataset %s before resampling w/ size: %s and number of classes: %s---> %s' % (
            dataset_name[1:], dataset.data.shape, n_classes, cls))

        # discriminating classes of the whole dataset
        dataset_train = ds.Dataset(dataset.data, dataset.target)
        dataset_train.separateSampleClass()
        data, target = dataset_train.getSampleClass()

        for i in xrange(0, n_classes):
            print ('\t\t#sample for class C%s: %s' % (i + 1, data[i].shape))
            if data[i].shape[0] < num_min_cls:
                num_min_cls = data[i].shape[0]

        resample = '/BALANCED'
        print ('%d.Class balancing...' % p_step)
        dataset.data, dataset.target = SMOTE(kind='regular', k_neighbors=num_min_cls - 1).fit_sample(dataset.data,
                                                                                                     dataset.target)
        p_step += 1
    else:
        resample = '/UNBALANCED'

    # shuffling data
    print ('\tShuffling data...')
    dataset.shufflingDataset()

    print ('\tDataset %s w/ size: %s and number of classes: %s---> %s' % (
    dataset_name[1:], dataset.data.shape, n_classes, cls))

    # discriminating classes the whole dataset
    dataset_train = ds.Dataset(dataset.data, dataset.target)
    dataset_train.separateSampleClass()
    data, target = dataset_train.getSampleClass()

    for i in xrange(0, n_classes):
        print ('\t\t#sample for class C%s: %s' % (i + 1, data[i].shape))

    # Max number of features to use
    max_num_feat = 300
    step = 1
    # max_num_feat = dataset.data.shape[1]

    if max_num_feat > dataset.data.shape[1]:
        max_num_feat = dataset.data.shape[1]

    alpha = 10 #regularizatio parameter (typically alpha in [2,50])

    params = {

        'SCBA':
         # the smaller is alpha the sparser is the C matrix (fewer representatives)
            {
                'alpha': alpha,
                'norm_type': 1,
                'max_iter': 3000,
                'thr': [10 ** -8],
                'type_indices': 'nrmInd',
                'normalize': False,
                'GPU': False,
                'device': 0,
                'PCA': False,
                'verbose': False,
                'step': 1,
                'affine': False,
            }
        # it's possible to add other FS methods by modifying the correct file
    }

    fs_model = fs.FeatureSelector(name='SCBA', tp='SLB', params=params['SCBA'])
    fs_name = 'SCBA'

    # CLASSIFIERS (it's possible to add other classifier methods by adding entries into this list)
    clf_name = [
        "SVM"
        # "Decision Tree",
        # "KNN"
    ]
    model = [
        SVC(kernel="linear")
        # DecisionTreeClassifier(max_depth=5),
        # KNeighborsClassifier(n_neighbors=1)
    ]

    '''Perform K-fold Cross Validation...'''
    k_fold = 10

    #defining result folders
    fs_path_output = '/CSFS/FS/K_FOLD'
    checkFolder(path_data_folder, fs_path_output)

    res_path_output = '/CSFS/RESULTS/K_FOLD'
    checkFolder(path_data_folder, fs_path_output)

    all_scores = {}
    all_scores.update({fs_name: []})

    cc_fold = 0
    conf_dataset = {}

    X = dataset.data
    y = dataset.target
    kf = KFold(n_splits=k_fold)

    print ('%d.Running the Intra-Class-Specific Feature Selection and building the ensemble classifier...\n' % p_step)
    p_step += 1
    for train_index, test_index in kf.split(X):

        X_train_kth, X_test_kth = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print ('\tDOING %s-CROSS VALIDATION W/ TRAINING SET SIZE: %s' % (cc_fold + 1, X_train_kth.shape))

        ''' For the training data in each class we find the representative features and use them as a best subset feature
            (in representing each class sample) to perform classification
        '''

        csfs_res = {}

        for i in xrange(0, n_classes):
            cls_res = {
                'C' + str(cls[i]): {}
            }
            csfs_res.update(cls_res)

        kth_scores = {}
        for i in xrange(0, len(clf_name)):
            kth_scores.update({clf_name[i]: []})

        # check whether the 'curr_res_fs_fold' directory exists, otherwise create it
        curr_res_fs_fold = path_data_folder + '/' + fs_path_output + '/' + fs_name + resample
        checkFolder(path_data_folder, fs_path_output + '/' + fs_name + resample)

        # discriminating classes for the k-th fold of the training set
        data_train = ds.Dataset(X_train_kth, y_train)
        data_train.separateSampleClass()
        ktrain_data, ktrain_target = data_train.getSampleClass()
        K_cls_ind_train = data_train.ind_class

        for i in xrange(0, n_classes):
            # print ('Train set size C' + str(i + 1) + ':', ktrain_data[i].shape)

            print ('\tPerforming feature selection on class %d with shape %s' % (cls[i] + 1, ktrain_data[i].shape))

            start_time = time.time()
            idx = fs_model.fit(ktrain_data[i], ktrain_target[i])

            # print idx

            print('\tTotal Time = %s seconds\n' % (time.time() - start_time))

            csfs_res['C' + str(cls[i])]['idx'] = idx
            csfs_res['C' + str(cls[i])]['params'] = params[fs_name]

            # with open(curr_res_fs_fold + '/' + str(cc_fold + 1) + '-fold' + '.pickle', 'wb') as handle:
            #     pickle.dump(csfs_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

        ens_class = {}
        # learning a classifier (ccn) for each subset of 'n_rep' feature
        for j in xrange(0, max_num_feat):
            n_rep = j + 1  # first n_rep indices

            for i in xrange(0, n_classes):
                # get subset of feature from the i-th class
                idx = csfs_res['C' + str(cls[i])]['idx']

                # print idx[0:n_rep]

                X_train_fs = X_train_kth[:, idx[0:n_rep]]

                _clf = i_clf.Classifier(names=clf_name, classifiers=model)
                _clf.train(X_train_fs, y_train)

                csfs_res['C' + str(cls[i])]['accuracy'] = _clf.classify(X_test_kth[:, idx[0:n_rep]], y_test)

            DTS = classificationDecisionRule(csfs_res, cls, clf_name, y_test)

            for i in xrange(0, len(clf_name)):
                _score = DTS[clf_name[i]]
                # print ('Accuracy w/ %d feature: %f' % (n_rep, _score))
                kth_scores[clf_name[i]].append(_score)

        x = np.arange(1, max_num_feat + 1)

        kth_results = {
            'clf_name': clf_name,
            'x': x,
            'scores': kth_scores,
        }

        all_scores[fs_name].append(kth_results)

        # saving k-th dataset configuration
        # with open(path_data_folder + fs_path_output + '/' + str(cc_fold + 1) + '-fold_conf_dataset.pickle',
        #           'wb') as handle:  # TODO: customize output name for recognizing FS parameters' method
        #     pickle.dump(conf_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cc_fold += 1

    # print all_scores

    print('%s.Averaging results...\n' % p_step)
    p_step += 1
    # Averaging results on k-fold

    # check whether the 'curr_res_fs_fold' directory exists, otherwise create it
    curr_res_output_fold = path_data_folder + '/' + res_path_output + '/' + fs_name + resample
    checkFolder(path_data_folder, res_path_output + '/' + fs_name + resample)

    M = {}
    for i in xrange(0, len(clf_name)):
        M.update({clf_name[i]: np.ones([k_fold, max_num_feat]) * 0})

    avg_scores = {}
    std_scores = {}
    for i in xrange(0, len(clf_name)):
        avg_scores.update({clf_name[i]: []})
        std_scores.update({clf_name[i]: []})

    # k-fold results for each classifier
    for k in xrange(0, k_fold):
        for clf in clf_name:
            M[clf][k, :] = all_scores[fs_name][k]['scores'][clf][:max_num_feat]

    for clf in clf_name:
        avg_scores[clf] = np.mean(M[clf], axis=0)
        std_scores[clf] = np.std(M[clf], axis=0)

    x = np.arange(1, max_num_feat + 1)
    results = {
        'clf_name': clf_name,
        'x': x,
        'M': M,
        'scores': avg_scores,
        'std': std_scores
    }

    # print avg_scores

    with open(curr_res_output_fold + '/clf_results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print ('Done with %s, [%d-cross validation] ' % (dataset_name[1:], k_fold))


if __name__ == '__main__':
    main()