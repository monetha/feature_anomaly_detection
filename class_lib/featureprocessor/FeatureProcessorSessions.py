from .FeatureProcessorBase import *

class FeatureProcessorSessions(FeatureProcessorBase):
    
    def __init__(self, prefix):
        super().__init__(prefix)
        self.quantilies = {}
    def make_session_day_part(self, df):

        df['session_start'] = pd.to_datetime(df['session_start'])
        df['session_end'] = pd.to_datetime(df['session_end'])


        df['session_length'] = df['session_end'] - df['session_start']
        df['session_length'] = df['session_length'].apply(lambda x: x.total_seconds())

        df['session_day_part'] = df['session_start'].apply(lambda x: 'morning' 
                if 6 <= x.hour <= 11
                else
                "afternoon" if 12 <= x.hour <= 17
                else
                "evening" if 18 <= x.hour <= 21
                else
                "night")
        
        return df
    
    def __time_since(self, x, name):
        try:
            valuable_events = x[x['name'] == name]
            valuable_events['time_'] = valuable_events['event_time'] - valuable_events['session_start']
            total_seconds = valuable_events.sort_values('time_')['time_'].iloc[0].total_seconds()
            if total_seconds < 0:
                return 0
            else:
                return total_seconds
        except:
            return np.nan
        



    def make_quntilies(self, sessions, session_id, valuable_event_names, events= None, events_session_id = None, rename = True, quatilies_formed = {}):
        session_features = sessions
        if events is not None:
            session_features = events.merge(sessions,
                                    left_on=events_session_id, 
                                    right_on=session_id,
                                    how ="left",
                                    suffixes = ('', "drop")
                                   )
        
        calc_vals = ['event_time','session_start', 'name']

        session_time_since_valuable = pd.DataFrame(session_features[session_id].unique(),
                                                   columns=[session_id])
        for name_ in valuable_event_names:
            name  = f'seconds_to_{name_}'
            grouped = session_features.groupby(session_id)[calc_vals].apply(self.__time_since,name= name_).reset_index()
            grouped = grouped.rename(columns = {0:name})
            session_time_since_valuable = session_time_since_valuable.merge(grouped, on=session_id, how = 'left')


        for name in [f'seconds_to_{temp_name}' for temp_name in valuable_event_names]:
            
            quantilies = quatilies_formed.get(name,None)
            if quantilies == None:
                quantilies = [session_time_since_valuable[name].quantile(0)]

                for i in [.25,.5,.75]:
                    quantilies.append(session_time_since_valuable[name].quantile(i))
                quantilies.append(session_time_since_valuable[name].quantile(1.0))
                print(quantilies)
                self.quantilies[name] = quantilies
            
            session_time_since_valuable[name]=session_time_since_valuable[name].apply(self._quantilies_filler, 
                                                                                          quantilies = quantilies)
        if rename:
            session_time_since_valuable.columns = [session_id] + list(self._prefix + session_time_since_valuable.columns[1:])
        sessions = sessions.merge(
            session_time_since_valuable,
            left_on= session_id,
            right_on=session_id,
            how='left'
        )
        return sessions
        