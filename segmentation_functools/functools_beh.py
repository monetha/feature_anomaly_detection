import pandas as pd

def get_mark_atb(df):
    df = df.sort_values(by=['session_start'])
    first_session_id = df[df['session_id'] == df.iloc[0]['session_id']]
    if any(first_session_id['name'].str.contains('add_to_basket')):
        return 2
    df = df[df.session_id != first_session_id.iloc[0]['session_id']]
    if any(df['name'].str.contains('add_to_basket')):
        return 1
    else:
        return 0
    
def get_marks(df):
    hs = 1 if df.session_id.nunique() > 1 else 0 
    hatb = get_mark_atb(df)
    return pd.DataFrame({
        'guest_id' : [df.iloc[0]['guest_id']],
        'had_add_to_basket' : [hatb],
        'had_second_session' : [hs]
                })

def get_events_before_ATB(df, col, start, stop):
    df = df.sort_values(by=['session_start', 'event_time'])
    mask = ~(df[col]==start).cummax() ^ (df[col]==stop).cummax()
    return df[mask]

def get_first_sessions(df):
    df.sort_values(by=['session_start'])
    return df[df['id'] == df.iloc[0]['id']]


def get_add_to_basket(df):
    if 'add_to_basket' in df['name'].unique():
        return 1
    return 0 

