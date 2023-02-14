import pandas as pd

class FeatureMinorizer:

    def __init__(self, minorisation_percentage_threshold, max_percentage_obs):
        self.minorisation_percentage_threshold = minorisation_percentage_threshold
        self.max_percentage_obs = max_percentage_obs
        
        
    def inplace_imputation(self,df,cat_features, float_features):
        for f in cat_features:
            df[f] = df[f].astype(str)
            df[f] = df[f].fillna('NaN')
        for f in float_features:
            df[f] = df[f].astype(float)
            df[f] = df[f].fillna(df[f].mean())    
        return (df)
    
    def minorise_feature(self, df, f, ths, replace_int=False):
        f_counts = pd.DataFrame(df[f].value_counts().reset_index())
        minors = f_counts['index'][f_counts[f]<ths]
        if replace_int:
            df.loc[df[f].isin(minors),f] = 0
        else:
            df.loc[df[f].isin(minors),f] = 'Other'
        return(df)
    
    def calculate_ths(self, column,shape):
        percentage_sum = 0
        unique_values = sorted(list(set(column)))
        for i, per_val in enumerate(column):
            percentage_sum += per_val/shape
            if percentage_sum > self.minorisation_percentage_threshold and per_val/shape > self.max_percentage_obs:
                index = unique_values.index(per_val)
                return unique_values[0] if index == 0 else unique_values[index-1]
        return 0

    def inplace_minorization(self, df, cat_features, ignore_features_list):

        minorization_ths = []
        for i in cat_features:
            if not any([1 if j  in i else 0 for j in ignore_features_list]):
                ths = self.calculate_ths(df[i].value_counts(ascending=True), df.shape[0])
                minorization_ths.append((ths, i))

        for ths, column_name in minorization_ths:
            df = self.minorise_feature(df, column_name, ths)
        return df