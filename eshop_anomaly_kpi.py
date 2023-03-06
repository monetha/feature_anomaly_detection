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
parser.add_argument('--to_sql', action='store_true')
parser.add_argument('--no_sql', dest='to_sql', action='store_false')
parser.set_defaults(to_sql=False)
args = parser.parse_args()

# ACCOUNT_ID = args.account_id
# SLICE = args.slice
# BEG_DATE = args.beg_date
# END_DATE = args.end_date
# LQ = args.lq
# RQ = args.rq
to_sql = args.to_sql

TIME = int(time.time())

BEG_DATE = datetime.datetime.strptime(BEG_DATE, '%Y-%m-%d').date() - timedelta(days=1)
END_DATE = datetime.datetime.strptime(END_DATE, '%Y-%m-%d').date()
LQ = LQ/100
RQ = RQ/100
anomaly_border = AN_BORD

print('Arguments accepted', 
      '\n Acc_id: ', ACCOUNT_ID,  
      '\n B_date:', BEG_DATE,
      '\n E_date:', END_DATE,
      '\n lq: ', LQ,
      '\n rq: ', RQ,
      '\n anomaly_border: ', anomaly_border,
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
    and created >= '{BEG_DATE}' and created < '{END_DATE}'

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
print('Step 1...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

sessions['duration'] = sessions['session_end'] - sessions['session_start']
sessions['duration'] = sessions['duration'].apply(lambda x: x.total_seconds())

statuses = statuses.rename(columns={'id': 'journey_status_id'})
sources = sources.rename(columns={'id': 'source_id'})
mediums = mediums.rename(columns={'id': 'medium_id'})
campaigns = campaigns.rename(columns={'id': 'campaign_id'})

sessions1 = sessions1.merge(statuses[['journey_status_id', 'status_name']], how='left', on='journey_status_id')
sessions = sessions.merge(sources, how='left', on='source_id').merge(mediums, how='left', on='medium_id').merge(campaigns, how='left', on='campaign_id')
sessions = sessions.merge(sessions1[['id', 'journey_status_id', 'created', 'status_name']], how='inner', on='id')

chanls = ['Organic Search', 'Social', 'Direct', 'Referral', 'Email', 'Paid Search']

data_tables = []
now = BEG_DATE + timedelta(days=1)
then = now + timedelta(days=31)
window_dates = []

for dt in rrule.rrule(rrule.HOURLY, dtstart=now, until=then):
    row = sessions[(sessions['session_start'] > dt - timedelta(days=1)) & (sessions['session_start'] <= dt)]
    data_tables.append(row)
    window_dates.append([dt - timedelta(days=1), dt])
    

anomaly_table = pd.DataFrame(columns=['period', 'period_len', 'period_begin', 'period_end', 
                                      'first_session', 'last_session',
                               'bounce_rate', 'conversion_rate', 'med_duration',
                               'bounce_organic', 'bounce_social', 'bounce_direct', 'bounce_referral', 'bounce_email',
                               'bounce_paid','conversion_organic', 'conversion_social', 'conversion_direct',
                               'conversion_referral', 'conversion_email', 'conversion_paid',
                               'duration_organic', 'duration_social', 'duration_direct', 'duration_referral',
                               'duration_email', 'duration_paid'], 
                      index=[i for i in range(0, len(data_tables))])

per = 0
for table in data_tables:
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('period')] = per
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('first_session')] = table.session_start.min()
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('last_session')] = table.session_start.max()
    le = len(table)
    bounc = len(table[table.status_name == 'Bounce'])
    if le == 0:
        anomaly_table.iloc[per, anomaly_table.columns.get_loc('bounce_rate')] = 0
    else:
        anomaly_table.iloc[per, anomaly_table.columns.get_loc('bounce_rate')] = bounc/le
    atb = len(table[table.add_to_basket_count >= 1])
    if le == 0:
        anomaly_table.iloc[per, anomaly_table.columns.get_loc('conversion_rate')] = 0
    else:
        anomaly_table.iloc[per, anomaly_table.columns.get_loc('conversion_rate')] = atb/le
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('med_duration')] = table.duration.median()
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('period_len')] = len(table)
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('period_begin')] = window_dates[per][0]
    anomaly_table.iloc[per, anomaly_table.columns.get_loc('period_end')] = window_dates[per][1]
    
    per += 1

channel_shorts = ['organic', 'social', 'direct', 'referral', 'email', 'paid']
for n in channel_shorts:
    per = 0
    for table in data_tables:
        if n == channel_shorts[0]:
            cat = table[table.channel == 'Organic Search']
        elif n == channel_shorts[1]:
            cat = table[table.channel == 'Social']
        elif n == channel_shorts[2]:
            cat = table[table.channel == 'Direct']
        elif n == channel_shorts[3]:
            cat = table[table.channel == 'Referral']
        elif n == channel_shorts[4]:
            cat = table[table.channel == 'Email']
        elif n == channel_shorts[5]:
            cat = table[table.channel == 'Paid Search']
        le = len(cat)
        bounc = len(cat[cat.status_name == 'Bounce'])
        try:
            anomaly_table.iloc[per, anomaly_table.columns.get_loc('bounce_{0}'.format(n))] = bounc/le
        except ZeroDivisionError:
            anomaly_table.iloc[per, anomaly_table.columns.get_loc('bounce_{0}'.format(n))] = 0
        atb = len(cat[cat.add_to_basket_count >= 1])
        try:
            anomaly_table.iloc[per, anomaly_table.columns.get_loc('conversion_{0}'.format(n))] = atb/le
        except ZeroDivisionError:
            anomaly_table.iloc[per, anomaly_table.columns.get_loc('conversion_{0}'.format(n))] = 0
            
        if cat.duration.sum() != 0:   
            try:
                anomaly_table.iloc[per, anomaly_table.columns.get_loc('duration_{0}'.format(n))] = cat.duration.median()
            except ZeroDivisionError:
                anomaly_table.iloc[per, anomaly_table.columns.get_loc('duration_{0}'.format(n))] = 0
        else:
            anomaly_table.iloc[per, anomaly_table.columns.get_loc('duration_{0}'.format(n))] = 0
        
        per += 1
        
metlist = ['bounce_rate', 'conversion_rate', 'med_duration', 'bounce_organic', 'bounce_social', 'bounce_direct',
           'bounce_referral', 'bounce_email', 'bounce_paid', 'conversion_organic', 'conversion_social',
           'conversion_direct', 'conversion_referral', 'conversion_email', 'conversion_paid', 
           'duration_organic', 'duration_social', 'duration_direct', 'duration_referral',
           'duration_email', 'duration_paid']


qframe = pd.DataFrame(columns=['metric', 'bot_line', 'upp_line'], index=[i for i in range(0, len(metlist))])
metn = 0
for n in metlist:
    anomaly_table['an_{}'.format(n)] = 0
    q95 = np.quantile(anomaly_table[n], RQ)
    q05 = np.quantile(anomaly_table[n], LQ)
    anomaly_table.loc[(anomaly_table[n] > q95) | (anomaly_table[n] < q05), 'an_{}'.format(n)] = 1
    qframe.iloc[metn, qframe.columns.get_loc('metric')] = n
    qframe.iloc[metn, qframe.columns.get_loc('bot_line')] = q05
    qframe.iloc[metn, qframe.columns.get_loc('upp_line')] = q95
    
    metn += 1

# qframe.to_csv('metric_lines_{0}_{1}_{2}.csv'.format(ACCOUNT_ID, BEG_DATE, END_DATE))
qframe.to_csv('metric_lines.csv')

anomaly_table['anomaly_coeff']= anomaly_table.iloc[:, -21:-1].sum(axis=1)
# anomaly_table.to_csv('anomaly_table_{0}_{1}_{2}.csv'.format(ACCOUNT_ID, BEG_DATE, END_DATE))
anomaly_table.to_csv('anomaly_table.csv')

#-----------------------------------



# timestamp
print('Step 2...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)

sort_data = (anomaly_table[anomaly_table['anomaly_coeff'] >= anomaly_border].index.tolist())

data_tables_an_found = []
for i in sort_data:
    data_tables_an_found.append(data_tables[i])


smcid = [['source_id', 'medium_id', 'campaign_id', 'ipcountry', 'device_family']]
kpiv = ['add_to_basket_count', 'duration', 'status_name']

tops_table = pd.DataFrame()
period_number = 0

print('It might take longer...')

pbar = tqdm.tqdm(total=len(data_tables_an_found))

for sess in data_tables_an_found:
    period_number += 1
    sliced_sessions_data = sess[['source', 'medium', 'campaign', 'ipcountry', 'device_family', 'add_to_basket_count',
                       'duration', 'status_name']]
    combs = sliced_sessions_data.groupby(['source', 'medium', 'campaign', 'ipcountry',
                            'device_family']).size().reset_index().rename(columns={0: 'length'})
    
    combs['bounce_rate'] = 0
    combs['conversion_rate'] = 0
    combs['med_duration'] = 0
    
    rown = 0

    for index, row in combs.iterrows():

        cuts = sliced_sessions_data[(sliced_sessions_data.source == row.source) & 
                                    (sliced_sessions_data.medium == row.medium) & 
                                    (sliced_sessions_data.campaign == row.campaign) & 
                                    (sliced_sessions_data.ipcountry == row.ipcountry) &
                                    (sliced_sessions_data.device_family == row.device_family)]

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

        for i in range(0, len(kpi)):
            val = kpi[i]
            if val == 'bounce_rate':
                cut = category_slice.sort_values(val, ascending=True).head(1)
            else:
                cut = category_slice.sort_values(val, ascending=False).head(1)
            cut['target_kpi'] = val
            cut['kpi_value'] = cut[val]
            cut['kpi_norma'] = norm[val].iloc[0]
            cut['kpi_delta'] = (cut[val] - norm[val].iloc[0]) / norm[val].iloc[0] * 100
            tops = tops.append(cut)
    
    tops = tops.reset_index(drop=True)
    tops['period_number'] = period_number
    tops['first_session'] = sess['session_start'].min()
    tops['last_session'] = sess['session_start'].max()
    tops['overall_begin'] = BEG_DATE
    tops['overall_end'] = END_DATE
    tops_table = tops_table.append(tops)
    
    pbar.update(1)

pbar.close()

tops_table = tops_table.reset_index(drop=True)
# tops_table.to_csv('tops_kpi_{0}_{1}_{2}.csv'.format(ACCOUNT_ID, BEG_DATE, END_DATE))
tops_table.to_csv('tops_kpi.csv')

if to_sql == True:
    
    host1 = os.getenv('DATA_DB_HOST')
    db1 = os.getenv('DATA_DB_NAME')
    user1 = os.getenv('DATA_DB_USER')
    password1 = os.getenv('DATA_DB_PASSWORD')
    port1 = os.getenv('DATA_DB_PORT')

    connection_str1 = 'postgresql://{0}:{1}@{2}:{3}/{4}'.format(user1, password1, host1, port1, db1)
    engine1 = sqlalchemy.create_engine(connection_str1)
    
    tops_table.to_sql(name='eshop_anomaly_tops_kpi', con=engine1, schema='data', if_exists='append')
    anomaly_table.to_sql(name='eshop_anomaly_table', con=engine1, schema='data', if_exists='append')

# timestamp
print('Core procession is finished...')
TIME_END = time.time()
print('Time spent: ', TIME_END - TIME)







    