
import numpy as np
import pandas as pd
import pickle

import json
import os

class Loader(object):

    def __init__(self, load_path):

        self.load_path = load_path
        self.check_load_path()
    
    def check_load_path(self):

        if not os.path.exists(self.load_path):

            print('Error! Load path ({}) does not exist!'.format(self.load_path))
    
    def load(self, filename, **kwargs):

        raise NotImplementedError


class TsvLoader(Loader):

    def __init__(self, load_path):

        super(TsvLoader, self).__init__(load_path)
    
    def load(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        record = pd.read_csv(filename, **kwargs)

        return record

class NpyLoader(Loader):
    
    def __init__(self, load_path):
        
        super(NpyLoader, self).__init__(load_path)
    
    def load(self, filename, **kwargs):

        filename = os.path.join(self.load_path, filename)
        record = np.load(filename, **kwargs)

        return record

class JsonLoader(Loader):

    def __init__(self, load_path):

        super(JsonLoader, self).__init__(load_path)
    
    def load(self, filename, **kwargs):

        filename_ = os.path.join(self.load_path, filename)
        with open(filename_, 'r') as f:
            record = json.loads(f.read())
        
        return record

class PickleLoader(Loader):
    
    def __init__(self, load_path):

        super(PickleLoader, self).__init__(load_path)
    
    def load(self, filename, **kwargs):

        filename_ = os.path.join(self.load_path, filename)
        with open(filename_, 'rb') as f:
            record = pickle.load(f)
        
        return record
