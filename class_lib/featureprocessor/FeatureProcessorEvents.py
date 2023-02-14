from .FeatureProcessorBase import *
 
 
class FeatureProcessorEvents(FeatureProcessorBase):
    def __name_mf(self, df, cat_name, name):
        try:
            return df[cat_name].value_counts().loc[name]/df[cat_name].value_counts().sum()
        except:
            return np.nan
        
    def __init__(self, prefix):
        self._prefix = prefix + '_'
        super().__init__(prefix)
        
    def prepare_aggregated_features(self, df, groupby_id, float_cols, cat_cols, names):
        result_df = super().prepare_aggregated_features(df,groupby_id, float_cols, cat_cols)

        if not names:
            return (result_df)
            
        previous_slice_len = len(result_df.columns)
        gr = df.groupby(groupby_id)[names]
        
        for cat_name in names:
            cat_names = df[cat_name].unique()
            for name in cat_names:
                temp = gr.apply(self.__name_mf, cat_name, name).reset_index()
                temp.columns = [groupby_id] + [f'{cat_name}_{name}_weight'] 
                result_df = result_df.merge(temp,left_on=groupby_id, right_on=groupby_id)
        self.feature_list += list('events_' + result_df.columns[previous_slice_len:])
        result_df.columns =list(result_df.columns[:previous_slice_len]) + list(self._prefix  + result_df.columns[previous_slice_len:])
        self.float_feature_list += list(result_df.columns[previous_slice_len:])
        return(result_df)

    def prepare_weight_mf(self, df,groupby_id, names):
        result_df = pd.DataFrame({'guest_id' : df.guest_id.unique()})
        gr = df.groupby(groupby_id)[names]
        
        for cat_name in names:
            cat_names = df[cat_name].unique()
            for name in cat_names:
                temp = gr.apply(self.__name_mf, cat_name, name).reset_index()
                temp.columns = [groupby_id] + [f'{cat_name}_{name}_weight'] 
                result_df = result_df.merge(temp,left_on=groupby_id, right_on=groupby_id)

                self.float_feature_list += [f'{self._prefix}{cat_name}_{name}_weight'] 
            
        
        result_df.columns = [groupby_id] + [self._prefix + i for i in result_df.columns[1:]]
        return result_df
