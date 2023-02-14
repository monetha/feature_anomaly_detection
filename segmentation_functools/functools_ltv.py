import re

def count_ltv(x):
    atb = x[x['name'] == 'add_to_basket']
    sum_of_all = sum([float(re.findall(r"[-+]?\d*\.?\d+|\d+", i['params']['price'].replace(',','.'))[0]) \
        * float(i['params'].get('quantity',0)) for i in atb['event_data']])
    return sum_of_all

def count_first_purchase_value(x):
    row = x[x['name'] =='add_to_basket'].iloc[0].event_data
    return float(re.findall(r"[-+]?\d*\.?\d+|\d+", row['params']['price'].replace(',','.'))[0]) * float(row['params'].get('quantity',0))
    