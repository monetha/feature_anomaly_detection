import pandas as pd
import numpy as np
import os
import sqlalchemy
import time
import math

from collections import Counter
import plotly.express as pe
from dotenv import load_dotenv
# fix this inputs if needed
# import plotly.io as pio
# pio.renderers.default='notebook'

import datetime
from datetime import date
import copy

from dython import nominal

import argparse
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sqlalchemy import engine

from class_lib.feature_generator import *
from class_lib.segments_preparer import *
from class_lib.graph_lib import *
from class_lib.featureprocessor import *
from class_lib.feature_minorizer import *

from config import *
from batch_config import *
from conv_config import *

from segmentation_functools.functools_beh import *
from segmentation_functools.functools_ltv import *
from segmentation_functools.functools import *

try:
    load_dotenv(DOTENV_PATH)
except:    
    load_dotenv('.env')

import warnings
warnings.filterwarnings("ignore")
#-----------------------------------

# def valid_date(s):
#     try:
#         return datetime.strptime(s, "%Y-%m-%d")
#     except ValueError:
#         msg = "not a valid date: {0!r}".format(s)
#         raise argparse.ArgumentTypeError(msg)

parser = argparse.ArgumentParser()
# parser.add_argument('--account_id', type=int, required=True)
# parser.add_argument('--slice', type=int, required=True)
# parser.add_argument('--beg_date', type=str, required=True)
# parser.add_argument('--end_date', type=str, required=True)
# parser.add_argument('--lq', type=int, required=True)
# parser.add_argument('--rq', type=int, required=True)
# parser.add_argument('--to_sql', type=bool, required=False, default=False)
parser.add_argument('--to_sql', action='store_true')
parser.add_argument('--no_sql', dest='to_sql', action='store_false')
parser.set_defaults(to_sql=False)
args = parser.parse_args()
args = parser.parse_args()

# ACCOUNT_ID = args.account_id
# SLICE = args.slice
# BEG_DATE = args.beg_date
# END_DATE = args.end_date
# LQ = args.lq
# RQ = args.rq
to_sql = args.to_sql

TIME = int(time.time())

BEG_DATE = datetime.datetime.strptime(BEG_DATE, '%Y-%m-%d').date()
END_DATE = datetime.datetime.strptime(END_DATE, '%Y-%m-%d').date()
LQ = LQ/100
RQ = RQ/100

print('Arguments accepted', 
      '\n Acc_id: ', ACCOUNT_ID, 
      '\n Slice: ', SLICE, 
      '\n B_date:', BEG_DATE,
      '\n E_date:', END_DATE,
      '\n lq: ', LQ,
      '\n rq: ', RQ,
      '\n to_sql: ', to_sql,
      '\nCurrent time: ', TIME)

#-----------------------------------

host = os.getenv('EVENTS_DB_HOST')
db = os.getenv('EVENTS_DB_NAME')
user = os.getenv('EVENTS_DB_USER')
password = os.getenv('EVENTS_DB_PASSWORD')
port = os.getenv('EVENTS_DB_PORT')

connection_str = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(user, password, host, port, db)
engine = sqlalchemy.create_engine(connection_str, execution_options={"stream_results":True})

sessions_q = f'''
select * from data.customer_profile_sessions cps
where cps.account_id = {ACCOUNT_ID} and cps.session_start >= '{BEG_DATE}' and cps.session_start < '{END_DATE}'
'''
sessions = pd.read_sql(sessions_q,engine)

events_q = f'''
select * from data.customer_profile_actions cps
where cps.account_id = {ACCOUNT_ID} and cps.event_time >= '{BEG_DATE}' and cps.event_time < '{END_DATE}'
'''
actions = pd.read_sql(events_q,engine)

visits_q = f'''
select * from data.customer_profile_visits cps
where cps.account_id = {ACCOUNT_ID} and cps.event_time >= '{BEG_DATE}' and cps.event_time < '{END_DATE}'
'''
visits = pd.read_sql(visits_q,engine)

# timestamp
print('Done reading 1...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

visits['name'] = 'page_view'

events_columns = actions.columns
events = pd.concat([actions,visits[events_columns]])

sessions = sessions.merge(
    events[['guest_id','session_id']].drop_duplicates(),
    left_on='id',
    right_on='session_id',
    how='left'
)

sessions = sessions[(sessions['session_end'] - sessions['session_start']) >= np.timedelta64(1, 's')]
events = events[events.session_id.isin(sessions.id)]
events = events.sort_values(by=['event_time'])

events['number'] = events.groupby('guest_id').cumcount()
def get_atb(x):
    temp = x[x['name'] == 'add_to_basket']
    if temp.shape[0] != 0:
        return temp.iloc[0].number
    return x.shape[0] + 1
events_sliced = events.groupby('guest_id').apply(lambda x : x[x['number'] < get_atb(x)])

target_1 = list(events[events['name'] == 'add_to_basket'].guest_id.unique())
target_0 = list(set(events.guest_id.unique()) - set(target_1))
target = pd.Series(
    np.concatenate([np.ones(len(target_1)),np.zeros(len(target_0))]),
    index = target_1+ target_0
)

# timestamp
print('event_sliced - done...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

events_processor = FeatureProcessorEvents('events_before_atb')
events_sliced = events_sliced.drop(columns=['guest_id'])
events_sliced = events_processor.time_func(events_sliced.reset_index(), 'guest_id','event_time')
events_float_features = []
events_cat_features = ['name', 'channel', 'referer']
events_features = events_processor.prepare_aggregated_features(events_sliced,
                                            'guest_id',
                                            events_float_features,
                                            events_cat_features,
                                            []
                                                )

# timestamp
print('events_features - done...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

NAN_AGG_TYPE_AEB = 'ignore'
events_features_nan = events_processor.prepare_aggregated_features_nan(
    events_sliced,
    'guest_id',
    ['interval_between'],
    interval_nan_type = NAN_AGG_TYPE_AEB
)  

# timestamp
print('events_features_nan - done...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

first_sessions = sessions[sessions.id.isin(events_sliced.session_id)]
session_processor = FeatureProcessorSessions('sessions_before_atb')
first_sessions = session_processor.make_session_day_part(first_sessions)
first_sessions = session_processor.time_func(
                                        first_sessions, 
                                        'guest_id',
                                        ['session_start','session_end']
                                        )

sessions_float_features = ['session_length','actions_count','page_views_count','attention_score']
sessions_cat_features = ['browser_family', 'os_family', 'device_family',
                        'device_brand', 'device_model', 'channel_session', 'device_type',
                        'session_day_part','time_to_link_click']

link_clicks = events_sliced[events_sliced['name'] =='link_click'].groupby('session_id').head(1)
first_sessions = first_sessions.merge(
    link_clicks[['session_id','event_time']],
    left_on='id',
    right_on='session_id',
    how='left'
)
first_sessions['time_to_link_click'] = (first_sessions['event_time'] - first_sessions['session_start']).dt.total_seconds()
first_sessions.loc[first_sessions['time_to_link_click'] < 0, 'time_to_link_click'] = 0
first_sessions['time_to_link_click'] =  pd.qcut(first_sessions.time_to_link_click,q=4).astype('str')

document_mouse_enter = events_sliced[events_sliced['name'] == 'document_mouse_enter'].groupby('session_id').size()
document_mouse_out = events_sliced[events_sliced['name'] == 'document_mouse_out'].groupby('session_id').size()
dme_dmo = document_mouse_enter.rename('document_mouse_enter').to_frame().join(document_mouse_out.rename('document_mouse_out'),how='outer').fillna(0)
dme_dmo['attention_score'] = (dme_dmo['document_mouse_out'] - dme_dmo['document_mouse_enter']) * dme_dmo.max(axis=1)
first_sessions = first_sessions.merge(
    dme_dmo['attention_score'],
    left_on=['id'],
    right_index=True,
    how='left'
)

# timestamp
print('first_sessions and dme_dmo - done...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

page_views_count = events_sliced[events_sliced['name'] =='page_view'].groupby('session_id').size()
actions_count = events_sliced[events_sliced['name'] != 'page_view'].groupby('session_id').size()
first_atb_event_time = events[events['name'] == 'add_to_basket'].groupby('session_id').head(1)['event_time']
first_sessions = first_sessions.drop(columns=["actions_count",'page_views_count'])
first_sessions = first_sessions.set_index('id')
first_sessions = first_sessions.join(
    page_views_count.rename('page_views_count'),
    how='left'
)
first_sessions = first_sessions.join(
    actions_count.rename('actions_count'),
    how='left'
)
first_sessions = first_sessions.join(
    first_atb_event_time.rename('first_atb_event_time'),
    how='left'
)

atb_sessions = first_sessions.loc[~first_sessions['first_atb_event_time'].isna()]
first_sessions.loc[~first_sessions['first_atb_event_time'].isna(),'session_length'] =\
atb_sessions['first_atb_event_time'] - atb_sessions['session_start']
first_sessions = first_sessions.rename(columns={'channel' : 'channel_session'})
sessions_features = session_processor.prepare_aggregated_features(
        first_sessions,
        'guest_id',
        sessions_float_features,
        sessions_cat_features
    )

# so as session_features_nan and events_features_nan arn't constant
try:
    merged = sessions_features.merge(events_features, left_on='guest_id', right_on='guest_id')\
    .merge(events_features_nan, left_on='guest_id', right_on='guest_id')\
    .merge(session_features_nan, left_on='guest_id', right_on='guest_id')
except:
    merged = sessions_features.merge(events_features, left_on='guest_id', right_on='guest_id')\
    .merge(events_features_nan, left_on='guest_id', right_on='guest_id')

# timestamp
print('Feature procession done...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)
#-----------------------------------
# this part will do as slow as big of a dataset and its chunks you get
query_sessions = f'''
    select 
    id as beh_id,
    guest_id,
    last_session_id,
    session_start as last_session_start,
    session_end as last_session_end
    from data.customer_profile_behaviour cpb 

    left join(
    select id as ses_id, session_start, session_end from data.customer_profile_sessions cps
    where cps.account_id = {ACCOUNT_ID}
    ) cps on cpb.last_session_id = cps.ses_id


    where cpb.account_id = {ACCOUNT_ID} and cps.session_start >= '{BEG_DATE}' and cps.session_start < '{END_DATE}' 
'''

newd = pd.read_sql_query(query_sessions, engine)
newd = newd.sort_values('last_session_start').reset_index(drop=True)
newd1 = newd[['last_session_id', 'last_session_start']]

# timestamp
print('Done reading 2...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

query_sessions = f'''
    select *
    from data.customer_profile_sessions cps 

    left join(
    select id as beh_id,guest_id, customer_profile_id from data.customer_profile_behaviour cpb
    where cpb.account_id = {ACCOUNT_ID}
    ) cpb on cpb.beh_id = cps.customer_profile_behaviour_id


    where cps.account_id = {ACCOUNT_ID} and cps.garbage_session = False and cps.session_start >= '{BEG_DATE}' and cps.session_start < '{END_DATE}' 
'''

sessions = pd.read_sql_query(query_sessions, engine)

# timestamp
print('Done reading 3...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

sessions = sessions[sessions.id.isin(newd.last_session_id)]
sessions = sessions.merge(newd1, how='inner', left_on='id', right_on='last_session_id').drop('last_session_id', axis=1).merge(
merged, how='inner', left_on='guest_id', right_on='guest_id')
sessions['screen'] = sessions['screen'].astype('str')
sessions = sessions.sort_values('session_start')

def split_df_chunks(data_df,chunk_size):
    total_length     = len(data_df)
    total_chunk_num  = math.ceil(total_length/chunk_size)
    normal_chunk_num = math.floor(total_length/chunk_size)
    chunks = []
    for i in range(normal_chunk_num):
        chunk = data_df[(i*chunk_size):((i+1)*chunk_size)]
        chunks.append(chunk)
    if total_chunk_num > normal_chunk_num:
        chunk = data_df[(normal_chunk_num*chunk_size):total_length]
        chunks.append(chunk)
    return chunks

data = split_df_chunks(sessions, SLICE)
features_to_drop = [
 'id',
 's_id',
 'account_id',
 'customer_profile_behaviour_id',
 'utm_source',
 'utm_medium',
 'utm_campaign',
 'utm_content',
 'utm_term',
 'ip',
 'session_start',
 'session_end',
 'created',
 'updated',
 's_ci',
 'source_id',
 'medium_id',
 'beh_id',
 'guest_id',
 'customer_profile_id',
 'last_session_start']

data1 = []
for d in data:
    data1.append(d.drop(features_to_drop, axis=1))

week_min = 0
week_max = len(data1) - 1
l = len(data1[0].columns)

# timestamp
print('Cor procession begin...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

# create corr matrix
dfc = []
for i in data1:
    assc = nominal.associations(i, compute_only=True, nom_nom_assoc='cramer', num_num_assoc='pearson', nom_num_assoc='correlation_ratio', mark_columns=True)
    dfc.append(assc['corr'])

# timestamp
print('Cor procession done...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)
    
# create dummy - shape for feature-feature corrs
def create_lofl(x, y, n=[]):
    lofl = []
    for i in range(y):
        ofl = []
        for j in range(x):
            ofl.append(n)
        lofl.append(ofl)
    return lofl

dum_list = create_lofl(l, l, [0])
lists = copy.deepcopy(dum_list)

for d in dfc:
    li = d.values.tolist()
    for i in range(0, l):
        for j in range(0, l):
            lists[i][j] = lists[i][j] + [li[i][j]]
for i in range(l):
        for j in range(l):
            lists[i][j] = lists[i][j][1:]

qlists = copy.deepcopy(lists)
q1 = 0
q9 = 0
for m in range(0, l):
        for b in range(0, l):
            q1 = np.quantile(qlists[m][b], LQ)
            q9 = np.quantile(qlists[m][b], RQ)
            qlists[m][b] = [q1, q9]

anlist = copy.deepcopy(dum_list)
listsc = copy.deepcopy(lists)
qlistc = copy.deepcopy(qlists)

# for values out of quantile range
for i in range(l):
        for j in range(l):
            it = 0
            for k in listsc[i][j]:
                it += 1
                if ((k < qlistc[i][j][0]) | (k > qlistc[i][j][1])):
                    anlist[i][j] = anlist[i][j] + [it]
for i in range(l):
        for j in range(l):
            anlist[i][j] = anlist[i][j][1:]

dfanlist = pd.DataFrame(anlist,index=dfc[0].columns,columns=dfc[0].columns)
dfanlist.to_csv('anomaly_matrix.csv') 

k = []
for i in range(0, l):
    for j in range(0, l):
        k += anlist[i][j]

data_dates1 = []
data_dates2 = []
for d in data:
    data_dates1.append(d['last_session_start'].min())
    data_dates2.append(d['last_session_start'].max())

k1 = pd.DataFrame(k, columns=['period']).value_counts().reset_index().rename(columns={0: 'counts'})
k1['counts'] = k1['counts'] / 2
k1['account_id'] = ACCOUNT_ID
k1['batch_size'] = SLICE
k1['feature_len'] = l
k1 = k1.sort_values('period')
k1['beg_date'] = data_dates1
k1['end_date'] = data_dates2
k1 = k1.sort_values('counts', ascending=False).reset_index(drop=True)
k1.to_csv('batch_counts.csv')

if to_sql == True:
    
    host1 = os.getenv('DATA_DB_HOST')
    db1 = os.getenv('DATA_DB_NAME')
    user1 = os.getenv('DATA_DB_USER')
    password1 = os.getenv('DATA_DB_PASSWORD')
    port1 = os.getenv('DATA_DB_PORT')

    connection_str1 = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(user1, password1, host1, port1, db1)
    engine1 = sqlalchemy.create_engine(connection_str1)
    
    k1.to_sql(name='feature_anomaly_batch_counts', con=engine1, schema='data', if_exists='append')

# timestamp
print('Cor procession done...')
print('Periods: ', week_max + 1)
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)







    