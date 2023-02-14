import pickle

import numpy as np

class FeatureSaver:
    def __init__(self, path):
        self.data = {}
        self.path = path
        
    def form_data(self,model_type : str, fg_names : list, fg_cat_names : list, fg_float_names : list):
        self.data[model_type] = {
            'names' : fg_names,
            'cat_names' : fg_cat_names,
            'float_names' : fg_float_names
        }
        
    def form_quatilies(self, data_type : str, quantilies : dict) -> dict:
        self.data[data_type] = {"quatilies" :quantilies}
      
    def get_data(self,model_type : str) -> dict:
        return self.data[model_type]
    
    def get_quantilies(self, data_type : str)-> dict:    
        return self.data[data_type]
    
    def save(self):
        with open(self.path +'/features.pkl', 'wb') as f:
            pickle.dump(self.data,f)
            
    def load(self):
        with open(self.path +'/features.pkl', 'rb') as f:
            self.data=pickle.load(f)
            
    def get_x_names(self, model_type : str ) -> np.ndarray:
        return self.data[model_type]['names']