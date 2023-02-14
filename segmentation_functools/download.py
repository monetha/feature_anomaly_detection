import pandas as pd
import logging


def download_to_file(account_id,engine,path): 
    print('asd')   
    logging.info('Downloading sessions')
    query_sessions = """
    select * from data.customer_profile_sessions cps
    where cps.account_id = {0}
    """.format(account_id)
    sessions = pd.read_sql_query(query_sessions, engine)
    sessions.to_csv(f'{path}/sessions.csv')
    logging.info('Sessions done')
    print('done')

    query_events = """
        select 
        cpa.session_id, 
        cpa.guest_id,
        cpa.channel,
        cpa.event_time,
        cpa.created,
        cpa.referer,
        cpa.url,
        cpa.name,
        cpa.event_data
        from data.customer_profile_actions cpa
        where cpa.account_id = {0}
        """.format(account_id)
    query_events_page_view = """
        select 
        cpa.session_id, 
        cpa.guest_id,
        cpa.channel,
        cpa.event_time,
        cpa.created,
        cpa.referer,
        cpa.url
        from data.customer_profile_visits cpa
        where cpa.account_id = {0}
        """.format(account_id)
        
    with open(f'{path}/events.csv', 'w') as f:
        last_i = 0
        logging.info('Events started')
        for i, partial_df in enumerate(pd.read_sql(query_events, engine, chunksize=100000)):
            partial_df.to_csv(f, index=False, header=(i == 0))
            last_i = i
        logging.info('Events done')
        logging.info('Visits started')
        for i, partial_df in enumerate(pd.read_sql(query_events_page_view, engine, chunksize=100000),last_i):
            partial_df['name'] = 'page_view'
            partial_df['event_data'] = '{}'
            partial_df.to_csv(f, index=False, header=0)         
        logging.info('Visits done')
    logging.info(f'Downloading done')