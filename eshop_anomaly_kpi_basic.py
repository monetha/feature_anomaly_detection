import pandas as pd
import numpy as np
import os
import sqlalchemy
import time
import math
import tqdm

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

from config import *
from kpi_config import *

from dateutil import rrule
from datetime import timedelta

try:
    load_dotenv(DOTENV_PATH)
except:    
    load_dotenv('.env')

import warnings
warnings.filterwarnings("ignore")
#-----------------------------------

parser = argparse.ArgumentParser()
# parser.add_argument('--account_id', type=int, required=True)
# parser.add_argument('--slice', type=int, required=True)
# parser.add_argument('--beg_date', type=str, required=True)
# parser.add_argument('--end_date', type=str, required=True)
# parser.add_argument('--lq', type=int, required=True)
# parser.add_argument('--rq', type=int, required=True)
# parser.add_argument('--to_sql', type=int, required=False, default=0)
args = parser.parse_args()

# ACCOUNT_ID = args.account_id
# SLICE = args.slice
# BEG_DATE = args.beg_date
# END_DATE = args.end_date
# LQ = args.lq
# RQ = args.rq

TIME = int(time.time())

END_DATE = datetime.datetime.now().date()
BEG_DATE = END_DATE - timedelta(days=7)
LQ = LQ/100
RQ = RQ/100
anomaly_border = 0

print('Arguments accepted', 
      '\n Acc_id: ', ACCOUNT_ID,  
      '\n B_date:', BEG_DATE,
      '\n E_date:', END_DATE,
      '\n lq: ', LQ,
      '\n rq: ', RQ,
      '\n anomaly_border: ', anomaly_border,
      '\nCurrent time: ', TIME)

#-----------------------------------

host = os.getenv('EVENTS_DB_HOST')
db = os.getenv('EVENTS_DB_NAME')
user = os.getenv('EVENTS_DB_USER')
password = os.getenv('EVENTS_DB_PASSWORD')
port = os.getenv('EVENTS_DB_PORT')
connection_str = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(user, password, host, port, db)

engine = sqlalchemy.create_engine(connection_str, execution_options={"stream_results":True})

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


query_sessions1 = f'''
    select *
    from data.customer_profile_session_journey_statuses
    where account_id = {ACCOUNT_ID} 
    and created >= '{BEG_DATE - timedelta(days=5)}' and created < '{END_DATE + timedelta(days=5)}'

'''
sessions1 = pd.read_sql_query(query_sessions1, engine)

query_sessions2 = f'''
    select *
    from data.customer_journey_statuses
    
'''
statuses = pd.read_sql_query(query_sessions2, engine)

query_sessions3 = f'''
    select *
    from data.sessions_campaign_dict
    
'''
campaigns = pd.read_sql_query(query_sessions3, engine)

query_sessions4 = f'''
    select *
    from data.sessions_source_dict
    
'''
sources = pd.read_sql_query(query_sessions4, engine)

query_sessions5 = f'''
    select *
    from data.sessions_medium_dict
    
'''
mediums = pd.read_sql_query(query_sessions5, engine)

# timestamp
# print('Step 1...')
# TIME_END = time.time()
# print('Time spent: ', TIME_END - TIME)

sessions['duration'] = sessions['session_end'] - sessions['session_start']
sessions['duration'] = sessions['duration'].apply(lambda x: x.total_seconds())

statuses = statuses.rename(columns={'id': 'journey_status_id'})
sources = sources.rename(columns={'id': 'source_id'})
mediums = mediums.rename(columns={'id': 'medium_id'})
campaigns = campaigns.rename(columns={'id': 'campaign_id'})

sessions1 = sessions1.merge(statuses[['journey_status_id', 'status_name']], how='left', on='journey_status_id')
sessions = sessions.merge(sources, how='left', on='source_id').merge(mediums, how='left', on='medium_id').merge(campaigns, how='left', on='campaign_id')
sessions = sessions.merge(sessions1[['id', 'journey_status_id', 'created', 'status_name']], how='inner', on='id')

# chanls = ['Organic Search', 'Social', 'Direct', 'Referral', 'Email', 'Paid Search']
chanls = sessions.channel.unique().tolist()

#-----------------------------------



# timestamp
# print('Step 2...')
# TIME_END = time.time()
# print('Time spent: ', TIME_END - TIME)



sess = sessions[['source', 'medium', 'campaign', 'ipcountry',
                        'device_type', 'add_to_basket_count', 'duration', 'status_name']]
if sess.empty:
    print('Error. No sessions')

smcid = [['source_id', 'medium_id', 'campaign_id', 'ipcountry', 'device_type']]
kpiv = ['add_to_basket_count', 'duration', 'status_name']

tops_table = pd.DataFrame()

sliced_sessions_data = sess[['source', 'medium', 'campaign', 'ipcountry', 'device_type', 'add_to_basket_count',
                       'duration', 'status_name']]
combs = sliced_sessions_data.groupby(['source', 'medium', 'campaign', 'ipcountry',
                        'device_type']).size().reset_index().rename(columns={0: 'length'})

combs['bounce_rate'] = 0
combs['conversion_rate'] = 0
combs['med_duration'] = 0

rown = 0

for index, row in combs.iterrows():

    cuts = sliced_sessions_data[(sliced_sessions_data.source == row.source) & 
                                (sliced_sessions_data.medium == row.medium) & 
                                (sliced_sessions_data.campaign == row.campaign) & 
                                (sliced_sessions_data.ipcountry == row.ipcountry) &
                                (sliced_sessions_data.device_type == row.device_type)]

    bounc = len(cuts[cuts.status_name == 'Bounce'])
    atb = len(cuts[cuts.add_to_basket_count >= 1])
    le = row['length']

    if bounc != 0:
        try:
            combs.iloc[rown, combs.columns.get_loc('bounce_rate')] = bounc/le
        except ZeroDivisionError:
            combs.iloc[rown, combs.columns.get_loc('bounce_rate')] = 0
    else:
        combs.iloc[rown, combs.columns.get_loc('bounce_rate')] = 0

    if atb != 0:
        try:
            combs.iloc[rown, combs.columns.get_loc('conversion_rate')] = atb/le
        except ZeroDivisionError:
            combs.iloc[rown, combs.columns.get_loc('conversion_rate')] = 0
    else:
        combs.iloc[rown, combs.columns.get_loc('conversion_rate')] = 0

    if cuts.duration.sum() != 0:   
        try:
            combs.iloc[rown, combs.columns.get_loc('med_duration')] = cuts.duration.median()
        except ZeroDivisionError:
            combs.iloc[rown, combs.columns.get_loc('med_duration')] = 0
    else:
        combs.iloc[rown, combs.columns.get_loc('med_duration')] = 0

    rown += 1

combs = combs[combs.length >= 10].reset_index(drop=True)
norms = sliced_sessions_data.groupby(['source']).size().reset_index().rename(columns={0: 'length'})
norms = norms[norms['source'].isin(combs.source.unique().tolist())].reset_index(drop=True)
norms['bounce_rate'] = 0
norms['conversion_rate'] = 0
norms['med_duration'] = 0

rown = 0

for index, row in norms.iterrows():

    cuts = sliced_sessions_data[(sliced_sessions_data.source == row.source)]

    bounc = len(cuts[cuts.status_name == 'Bounce'])
    atb = len(cuts[cuts.add_to_basket_count >= 1])
    le = row['length']

    if bounc != 0:
        try:
            norms.iloc[rown, norms.columns.get_loc('bounce_rate')] = bounc/le
        except ZeroDivisionError:
            norms.iloc[rown, norms.columns.get_loc('bounce_rate')] = 0
    else:
        norms.iloc[rown, norms.columns.get_loc('bounce_rate')] = 0

    if atb != 0:
        try:
            norms.iloc[rown, norms.columns.get_loc('conversion_rate')] = atb/le
        except ZeroDivisionError:
            norms.iloc[rown, norms.columns.get_loc('conversion_rate')] = 0
    else:
        norms.iloc[rown, norms.columns.get_loc('conversion_rate')] = 0

    if cuts.duration.sum() != 0:   
        try:
            norms.iloc[rown, norms.columns.get_loc('med_duration')] = cuts.duration.median()
        except ZeroDivisionError:
            norms.iloc[rown, norms.columns.get_loc('med_duration')] = 0
    else:
        norms.iloc[rown, norms.columns.get_loc('med_duration')] = 0

    rown += 1

kpi = ['bounce_rate', 'conversion_rate', 'med_duration']

tops = pd.DataFrame()

for sou in norms.source:

    category_slice = combs[combs.source == sou]
    norm = norms[norms.source == sou]

    if (len(category_slice[category_slice.length > 10]) >= 3):
        len_sllice = category_slice[category_slice.length > 10]
        for i in range(0, len(kpi)):
            val = kpi[i]
            if val == 'bounce_rate':
                cut = len_sllice.sort_values(val, ascending=True).head(5)
                cut2 = len_sllice.sort_values(val, ascending=False).head(5)
            else:
                cut = len_sllice.sort_values(val, ascending=False).head(5)
                cut2 = len_sllice.sort_values(val, ascending=True).head(5)
            cut['is_best'] = 1
            cut2['is_best'] = 0
            cut['target_kpi'] = val
            cut2['target_kpi'] = val
            cut['kpi_value'] = cut[val]
            cut2['kpi_value'] = cut2[val]
            cut['kpi_norma'] = norm[val].iloc[0]
            cut2['kpi_norma'] = norm[val].iloc[0]
            cut['kpi_delta'] = (cut[val] - norm[val].iloc[0]) / norm[val].iloc[0] * 100
            cut2['kpi_delta'] = (cut2[val] - norm[val].iloc[0]) / norm[val].iloc[0] * 100

            tops = tops.append(cut)
            tops = tops.append(cut2)

# print(len(tops))
try:
    if len(tops) != 0:
        tops = tops.reset_index(drop=True)
        tops['first_session'] = sess['session_start'].min()
        tops['last_session'] = sess['session_start'].max()
        tops['period_begin'] = sess['period_begin'].max()
        tops['period_end'] = sess['period_end'].max()
        tops['overall_begin'] = BEG_DATE
        tops['overall_end'] = END_DATE
except:
    print('__getitem__ problem has occured. currently working on it \n Results should be pretty normal though \n')
    tops = tops.reset_index(drop=True)
tops_table = tops_table.append(tops)

tops_table = tops_table.reset_index(drop=True)
# tops_table.to_csv('tops_kpi_{0}_{1}_{2}.csv'.format(ACCOUNT_ID, BEG_DATE, END_DATE))
tops_table.to_csv('tops_kpi.csv')


with open('cache_log.txt', 'w', encoding='utf-8') as f:
    f.write('Results with specs: \n')
    f.write( 
      'Acc_id: {0} \n B_date: {1} \n E_date: {2} \n lq: {3} \n rq: {4} \n anomaly_border: {5} \n\n'.format(
          ACCOUNT_ID,  
          BEG_DATE,
          END_DATE,
          LQ,
          RQ,
          anomaly_border))
    lenq = len(sess)
    f.write('Sessions overall: {0} \n\n'.format(lenq))
    if tops_table.empty:
            f.write('Insufficient data for MCID candidates evaluation')
    else:
        f.write('Best performing traffic \n\n')
        for sou in tops_table.source.unique():
            session_slice = sess[sess.source == sou]
            category_slice = tops_table[(tops_table.source == sou) & (tops_table.is_best == 1)]
            norm = norms[norms.source == sou]
            f.write('- {0} \n'.format(sou))
            f.write('- {0} % sessions with this source \n\n'.format(round(len(session_slice) / lenq * 100), 2))
            for index, row in category_slice.iterrows():
                    f.write('-- had {0}% traffic coming from {1}, {2} coming from {3}, {4} \n with target_kpi {5} \n'.format(round(row['length'] / len(session_slice) * 100, 2), row.medium, row.campaign, row.ipcountry, row.device_type, row.target_kpi))
                    f.write('--- bounce_rate ({2}) is {0} % diff from norm which is {1} \n'.format(round((row['bounce_rate'] - norm['bounce_rate'].iloc[0]) / norm['bounce_rate'].iloc[0] * 100, 2), round(norm['bounce_rate'].iloc[0], 2), round(row['bounce_rate'], 2)))
                    f.write('--- med_duration ({2}) is {0} % diff from norm which is {1} \n'.format(round((row['med_duration'] - norm['med_duration'].iloc[0]) / norm['med_duration'].iloc[0] * 100, 2), round(norm['med_duration'].iloc[0], 2), round(row['med_duration'], 2)))
                    f.write('--- conversion_rate ({2}) is {0} % diff from norm which is {1} \n'.format(round((row['conversion_rate'] - norm['conversion_rate'].iloc[0]) / norm['conversion_rate'].iloc[0] * 100, 2), round(norm['conversion_rate'].iloc[0], 2), round(row['conversion_rate'], 2)))
                    f.write('\n')
        
        f.write('Worst performing traffic \n\n')
        for sou in tops_table.source.unique():
            session_slice = sess[sess.source == sou]
            category_slice = tops_table[(tops_table.source == sou) & (tops_table.is_best == 0)]
            norm = norms[norms.source == sou]
            f.write('- {0} \n'.format(sou))
            f.write('- {0} % sessions with this source \n\n'.format(round(len(session_slice) / lenq * 100), 2))
            for index, row in category_slice.iterrows():
                    f.write('-- had {0}% traffic coming from {1}, {2} coming from {3}, {4} \n with target_kpi {5} \n'.format(round(row['length'] / len(session_slice) * 100, 2), row.medium, row.campaign, row.ipcountry, row.device_type, row.target_kpi))
                    f.write('--- bounce_rate ({2}) is {0} % diff from norm which is {1} \n'.format(round((row['bounce_rate'] - norm['bounce_rate'].iloc[0]) / norm['bounce_rate'].iloc[0] * 100, 2), round(norm['bounce_rate'].iloc[0], 2), round(row['bounce_rate'], 2)))
                    f.write('--- med_duration ({2}) is {0} % diff from norm which is {1} \n'.format(round((row['med_duration'] - norm['med_duration'].iloc[0]) / norm['med_duration'].iloc[0] * 100, 2), round(norm['med_duration'].iloc[0], 2), round(row['med_duration'], 2)))
                    f.write('--- conversion_rate ({2}) is {0} % diff from norm which is {1} \n'.format(round((row['conversion_rate'] - norm['conversion_rate'].iloc[0]) / norm['conversion_rate'].iloc[0] * 100, 2), round(norm['conversion_rate'].iloc[0], 2), round(row['conversion_rate'], 2)))
                    f.write('\n')


        
        
        
# timestamp
print('Core procession is finished...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)


# f.write('It was detected that for {0} sessions from {1} to {2} which originate from {3} and have the following MCID: {4}, {5}, {6}, {7}, the {8} was {9} and was equal to normal value in this period {10}.'.format(int(row['length']), row.period_begin, row.period_end, row.source, row.medium, row.campaign, row.ipcountry, row.device_family, row.target_kpi, round(row.kpi_value, 2), round(row.kpi_norma, 2))




    