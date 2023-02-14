from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

class CategoryRecoverer:
    '''
    Predicts category or subcategory on name and shop.
    1. It builds TF-IDF matrix for each shop
    2. It uses SVD truncate to reduce dimension of matrix.
    3. Classification is produced with random forest.
    '''
    def __init__(self):
        self.models = {}
        
    def TrainOneCat(self, data, model_idx, is_sub=False):
        '''
        Trains one group:
        
        data - dataset with name and name or subcategory
        model_idx - idx of model in dictionary of models
        is_sub - predicts subcategory or nit
        '''
        X = data['name']
        if is_sub:
            y = data['subcategory']
        else:
            y = data['category']
        self.models[model_idx] = {}
        
        min_df = int(max(X.shape[0] / 5e3, 1))
        if model_idx == 'total':
            min_df = 0.0005
                    
        vec = TfidfVectorizer(max_df=0.98, min_df=min_df)
        matrix = vec.fit_transform(X)
        self.models[model_idx]['vec'] = vec
        
        svd = TruncatedSVD(n_components=min(500, matrix.shape[1] - 1), random_state=123)
        features = svd.fit_transform(matrix)
        self.models[model_idx]['svd'] = svd
        
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=123)
        clf.fit(features, y)
        self.models[model_idx]['clf'] = clf
    
    def fit(self, data, is_sub=False):
        '''
        fits data:
        
        data - dataset with name and name or subcategory
        is_sub - predicts subcategory or not
        '''
        shops = set(data['eshop_id'].tolist())
        self.TrainOneCat(data, 'total', is_sub)
        for shop in shops:
            X = data[data['eshop_id'] == shop]
            if X.shape[0] == 1:
                continue
                
            # In subcategory task create more common model for just shop. It should increase guessed number
            self.TrainOneCat(X, str(shop))
            
            # At this block we create more specific models
            if is_sub:
                cats = set(data[data['eshop_id'] == shop]['category'].tolist())
                for cat in cats:
                    X = data[(data['eshop_id'] == shop) & (data['category'] == cat)]
                    if X.shape[0] <= 10:
                        continue
                    self.TrainOneCat(X, str(shop) + '_' + cat, is_sub)
                    
    def PredictOneGood(self, y, model_idx):
        '''
        predicts for one object
        
        y - name to predict
        model_idx - idx of model in dictionary of models
        '''
        model = self.models[model_idx]
        y = model['vec'].transform([y])
        y = model['svd'].transform(y)
        return model['clf'].predict(y)[0]
            
    
    def predict(self, y, is_sub=False):
        '''
        Predicts for dataset. Requires category and name columns
        
        y - name to predict
        model_idx - idx of model in dictionary of models
        '''
        preds = []
        for index, row in y.iterrows():
            model_idx = str(row['eshop_id'])
            
            # Check whether we are working with subcategories
            if is_sub:
                model_idx += '_' + row['category']
                
            # If no such shop, therefore there is no shop_category, so let's try to use all data
            if str(row['eshop_id']) not in self.models.keys():
                preds.append(self.PredictOneGood(row['name'], 'total'))
                
            # If there is shop_category
            elif model_idx in self.models.keys():
                preds.append(self.PredictOneGood(row['name'], model_idx))
                
            # We have a shop and don't have shop_category, let's use more wide model
            elif str(row['eshop_id']) in self.models.keys():
                preds.append(self.PredictOneGood(row['name'], str(row['eshop_id'])))
            else:
                preds.append(self.PredictOneGood(row['name'], 'total'))
        return preds