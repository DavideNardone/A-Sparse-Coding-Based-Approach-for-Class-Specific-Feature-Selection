import hdf5storage #dependency
import numpy as np

np.set_printoptions(threshold=np.inf)
import scipy.io as sio

class Loader:
    def __init__(self, file_path, name, variables, format, k_fold=None):


        # This Class provides several method for loading many type of dataset (matlab, csv, txt, etc)

        if format == 'matlab':  # classic workspace

            mc = sio.loadmat(file_path)

            for variable in variables:
                setattr(self, variable, mc[variable])

        elif format == 'matlab_struct':  # struct one level
            print ('Loading data...')

            mc = sio.loadmat(file_path)
            mc = mc[name][0, 0]

            for variable in variables:
                setattr(self, variable, mc[variable])

        elif format == 'custom_matlab':
            print ('Loading data...')

            mc = sio.loadmat(file_path)
            mc = mc[name][0, 0]

            for variable in variables:
                setattr(self, variable, mc[variable][0, 0])

        elif format == 'matlab_v73':
            mc = hdf5storage.loadmat(file_path)

            for variable in variables:
                setattr(self, variable, mc[variable])

    def getVariables(self, variables):

        D = {}

        for variable in variables:
            D[variable] = getattr(self, variable)

        return D