import numpy as np
np.set_printoptions(threshold=np.inf)
import sys
sys.path.insert(0, './src')
import SCBA as fs



class FeatureSelector:

    def __init__(self, model=None, name=None, tp=None, params=None):


        self.name = name
        self.model = model
        self.tp = tp
        self.params = params



    def fit(self, X, y):

        idx = []

        #add custom 'type' of Feature Selector
        if self.tp == 'filter':

            if self.name == 'Relief':
                '''
                add a custom Feature Selector such as:

                score = reliefF.reliefF(X, y)
                idx = reliefF.feature_ranking(score)

                '''

        elif self.tp == 'SLB':


            # SCBA method
            if self.name == 'GAD':

                alg = gd.GAD(X, self.params)
                _, idx = alg.iterative_GAD()

            if self.name == 'SCBA':
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

        return idx