import numpy as np
import pandas as pd

class FeatureProcessorBase():
    
    def __init__(self, prefix):
        self._prefix = prefix + '_'
        self.feature_list = []
        self.cat_feature_list = []
        self.float_feature_list = []
        self.interval_agg_functions = {
            'ignore' : self._ignore,
            'mean' : self._mean,
            'quantilies' : self._qantilies
        }
        self.__float_agg_func = [np.mean, np.sum, np.min, np.max]
        self.__cat_agg_func = [lambda x:x.value_counts().index[0],
                            lambda x:x.value_counts().iloc[0]/x.value_counts().sum(),
                            lambda x:x.iloc[0], 
                            lambda x:x.iloc[-1]]
        self.__cat_agg_func_names = [
            'mf',
            "weight_mf",
            'first', 
            'last']  

    def _quantilies_filler(self,x, quantilies):
        if np.isnan(x):
            return "NaN"
        for i in range(len(quantilies) - 1):
            if x >= quantilies[i] and x <=quantilies[i+1]:
                return f'{quantilies[i]}-{quantilies[i+1]}'
        return "NaN"
    
    def minorise_feature(self, df, f, ths, replace_int=False):
        f_counts = pd.DataFrame(df[f].value_counts().reset_index())
        minors = f_counts['index'][f_counts[f]<ths]
        if replace_int:
            df.loc[df[f].isin(minors),f] = 0
        else:
            df.loc[df[f].isin(minors),f] = 'Other'
        return(df)
    
    def inpace(self, df, cat_features, float_features):
        for f in cat_features:
            df[f] = df[f].astype(str)
            df[f] = df[f].fillna('NaN')
        for f in float_features:
            df[f] = df[f].astype(float)
            df[f] = df[f].fillna(df[f].mean())

        for i in cat_features:
            df = self.minorise_feature(df, i, self.minorisation_THS)
        return df
    
    def _ignore(self,df, groupby_id,col):
        temp = df.groupby(groupby_id)[[col]].agg(self.__float_agg_func).reset_index()
        temp.columns = [groupby_id] + ['_'.join(col).strip() for col in temp.columns.values[1:]]
        self.float_feature_list += list(temp.columns.values[1:])
        return temp
    
    def _mean(self,df, groupby_id,col):
        df[col].fillna((df[col].mean()), inplace=True)
        temp = df.groupby(groupby_id)[[col]].agg(self.__float_agg_func).reset_index()
        temp.columns = [groupby_id] + ['_'.join(col_).strip() for col_ in temp.columns.values[1:]]
        self.float_feature_list += list(temp.columns.values[1:])
        return temp
        
    def _qantilies(self,df , groupby_id,col):
             
        quantilies = [df[col].quantile(0)]

        for i in [.25,.5,.75]:
            quantilies.append(df[col].quantile(i))
        quantilies.append(df[col].quantile(1.0))
        df[col]=df[col].apply(self._quantilies_filler, quantilies = quantilies)
        temp =  df.groupby(groupby_id)[[col]].agg(self.__cat_agg_func).reset_index()
        temp.columns = [groupby_id] + [col + '_' + i  for i  in self.__cat_agg_func_names]
        self.cat_feature_list += list(temp.columns[1:])
        return temp
    
    def prepare_aggregated_features_nan(self, df, groupby_id, cols, interval_nan_type = None ):
        df = df[[groupby_id] + cols]
        result_df = pd.DataFrame(data={groupby_id: df[groupby_id].unique()})
        previos_cat_features = self.cat_feature_list
        previos_float_features = self.float_feature_list
        self.cat_feature_list = []
        self.float_feature_list = []
        for f in cols:
            aggregated_df = self.interval_agg_functions[interval_nan_type](df,groupby_id,f)
            result_df = result_df.merge(aggregated_df, left_on=groupby_id, right_on=groupby_id, how='left')

        result_df.columns = [groupby_id] + list(self._prefix + result_df.columns[1:])
        self.cat_feature_list = previos_cat_features + [self._prefix + i for i in self.cat_feature_list]
        self.float_feature_list = previos_float_features + [self._prefix + i for i in  self.float_feature_list]

        return (result_df)

    def prepare_aggregated_features(self, df, groupby_id, float_cols, cat_cols):
        cols = [float_cols, cat_cols]
        cols = [item for sublist in cols for item in sublist]
        cols.append(groupby_id)

        df = df[cols]
        df[cat_cols] = df[cat_cols].fillna('NaN')

        previos_cat_features = self.cat_feature_list
        previos_float_features = self.float_feature_list
        self.cat_feature_list = []
        self.float_feature_list = []
        

        cat_agg_df = df.groupby(groupby_id)[cat_cols].agg(self.__cat_agg_func).reset_index()
        cat_agg_df.columns = [groupby_id] + [i + '_' + j for i in cat_cols for j in self.__cat_agg_func_names]
        self.cat_feature_list += [i for i in cat_agg_df.columns[1:] if 'weight' not in i]
        self.float_feature_list += [i for i in cat_agg_df.columns[1:] if 'weight' in i]

        
        num_records_df = pd.DataFrame(df[groupby_id].value_counts()).reset_index()
        num_records_df.columns = [groupby_id, 'num_records']
        self.float_feature_list += ["num_records"]

        result_df = num_records_df.merge(cat_agg_df, left_on = groupby_id, right_on = groupby_id, how = 'left')
        if float_cols != []:
            float_agg_df = df.groupby(groupby_id)[float_cols].agg(self.__float_agg_func).reset_index()
            float_agg_df.columns = [groupby_id] + ['_'.join(col).strip() for col in float_agg_df.columns.values[1:]]
            result_df = result_df.merge(float_agg_df, left_on = groupby_id, right_on = groupby_id, how = 'left')
            self.float_feature_list += list(float_agg_df.columns[1:])
        result_df.columns = [groupby_id] + list(self._prefix + result_df.columns[1:])

        self.cat_feature_list =previos_cat_features + [self._prefix + i for i in self.cat_feature_list]
        self.float_feature_list =previos_float_features + [self._prefix + i for i in  self.float_feature_list]
        self.feature_list += list(result_df.columns[1:])
        return(result_df)
    
    def time_func(self, df,groupby_id,  field):
        return df.groupby(groupby_id).apply(self.get_intervals, field).reset_index(drop=True)
    
    def get_intervals(self, df, field):
        if isinstance(field, list):
            df = df.sort_values(by = [field[0]])
            df['interval_between'] = (df[field[0]] - df[field[1]].shift(1)).apply(lambda x: x.total_seconds())
        else:
            df = df.sort_values(by = [field])
            df['interval_between'] = df[field].diff().apply(lambda x: x.total_seconds())
        
        return df
