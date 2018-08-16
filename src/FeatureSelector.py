from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.feature_selection import mutual_info_classif

from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking
from skfeature.function.sparse_learning_based import RFS, ls_l21, ll_l21, MCFS, NDFS, UDFS

from skfeature.function.similarity_based import reliefF, fisher_score
from skfeature.function.information_theoretical_based import MRMR

import sys
sys.path.insert(0, './src')
import numpy as np
np.set_printoptions(threshold=np.inf)
import SMBA as fs


class FeatureSelector:

    def __init__(self, model=None, name=None, tp=None, params=None):

        self.name = name
        self.model = model
        self.tp = tp
        self.params = params

    def setParams(self, comb_par, params_name, params):

        for par_name, par in zip(params_name, comb_par):
            params[par_name] = par

        self.params = params

    def fit(self, X, y):

        idx = []

        if self.tp == 'ITB':

            if self.name == 'MRMR':
                idx = MRMR.mrmr(X, y, n_selected_features=self.params['num_feats'])

        elif self.tp == 'filter':

            if self.name == 'Relief':
                score = reliefF.reliefF(X, y, k=self.params['k'])
                idx = reliefF.feature_ranking(score)

            if self.name == 'Fisher':
                # obtain the score of each feature on the training set
                score = fisher_score.fisher_score(X, y)

                # rank features in descending order according to score
                idx = fisher_score.feature_ranking(score)

            if self.name == 'MI':
                idx = np.argsort(mutual_info_classif(X, y, n_neighbors=self.params['n_neighbors']))[::-1]

        elif self.tp == 'wrapper':

            model_fit = self.model.fit(X, y)
            model = SelectFromModel(model_fit, prefit=True)
            idx = model.get_support(indices=True)
        elif self.tp == 'SLB':

            # one-hot-encode on target
            y = construct_label_matrix(y)

            if self.name == 'SMBA':
                scba = fs.SCBA(data=X, alpha=self.params['alpha'], norm_type=self.params['norm_type'],
                               verbose=self.params['verbose'], thr=self.params['thr'], max_iter=self.params['max_iter'],
                               affine=self.params['affine'],
                               normalize=self.params['normalize'],
                               step=self.params['step'],
                               PCA=self.params['PCA'],
                               GPU=self.params['GPU'],
                               device = self.params['device'])

                nrmInd, sInd, repInd, _ = scba.admm()
                if self.params['type_indices'] == 'nrmInd':
                    idx = nrmInd
                elif self.params['type_indices'] == 'repInd':
                    idx = repInd
                else:
                    idx = sInd

            if self.name == 'RFS':
                W = RFS.rfs(X, y, gamma=self.params['gamma'])
                idx = feature_ranking(W)

            if self.name == 'll_l21':
                # obtain the feature weight matrix
                W, _, _ = ll_l21.proximal_gradient_descent(X, y, z=self.params['z'], verbose=False)
                # sort the feature scores in an ascending order according to the feature scores
                idx = feature_ranking(W)
            if self.name == 'ls_l21':
                # obtain the feature weight matrix
                W, _, _ = ls_l21.proximal_gradient_descent(X, y, z=self.params['z'], verbose=False)

                # sort the feature scores in an ascending order according to the feature scores
                idx = feature_ranking(W)

            if self.name == 'LASSO':

                LASSO = Lasso(alpha=self.params['alpha'], positive=True)

                y_pred_lasso = LASSO.fit(X, y)

                if y_pred_lasso.coef_.ndim == 1:
                    coeff = y_pred_lasso.coef_
                else:
                    coeff = np.asarray(y_pred_lasso.coef_[0, :])

                idx = np.argsort(-coeff)

            if self.name == 'EN':  # elastic net L1

                enet = ElasticNet(alpha=self.params['alpha'], l1_ratio=1, positive=True)
                y_pred_enet = enet.fit(X, y)

                if y_pred_enet.coef_.ndim == 1:
                    coeff = y_pred_enet.coef_
                else:
                    coeff = np.asarray(y_pred_enet.coef_[0, :])

                idx = np.argsort(-coeff)

        return idx