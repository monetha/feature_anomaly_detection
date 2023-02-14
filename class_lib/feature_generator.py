import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import itertools
import sys
from collections import Counter
import datetime

pd.options.mode.chained_assignment = None

def days_between_dates(data, data_first):
    x = datetime.datetime(data.year, data.month, data.day) - datetime.datetime(
        data_first.year, data_first.month, data_first.day)
    return x.days

def days_since_start(data, data_first):
    x = datetime.datetime(data.year, data.month, data.day) - datetime.datetime(data_first.year, data_first.month, 1)
    return x.days

def minorise_feature(df, f, ths, replace_int=False):
    f_counts = pd.DataFrame(df[f].value_counts().reset_index())
    minors = f_counts['index'][f_counts[f] < ths]
    if replace_int:
        df.loc[df[f].isin(minors), f] = 0
    else:
        df.loc[df[f].isin(minors), f] = 'Other'
    return (df)


class FeatureGenerator():
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.cat_names = []
        self.float_names = []
    def feature_encoder(self, feature):
        values = array(feature.values.astype(str))
        label_encoder = LabelEncoder()
        label_encoder.fit(values)
        int_encoded_values = label_encoder.fit_transform(values)
        int_encoded_values = int_encoded_values.reshape(len(int_encoded_values),1)
    
        onehot_encoder = OneHotEncoder(sparse = False)
        onehot_encoded_values = onehot_encoder.fit_transform(int_encoded_values)
        
        self.class_labels = label_encoder.classes_
        self.onehot_encoded = onehot_encoded_values
        
    def X_matrix_preparer(self, data, cat_cols = [], float_cols = [], isnull_cols = []):
        X = np.empty((data.shape[0],0))
        X_names = np.array([])

        if cat_cols != []:
            for c_i in cat_cols:    
                FG_i = FeatureGenerator(c_i)
                FG_i.feature_encoder(data[c_i])
                X = np.concatenate((X, FG_i.onehot_encoded), axis = 1)
                X_names = np.concatenate((X_names, [FG_i.feature_name + "_" + s for s in FG_i.class_labels]))
            self.cat_names = X_names
        if float_cols != []:
            for f_i in float_cols:
                f = data[f_i]
                f = np.array(f).reshape(len(f), 1)
                X = np.concatenate((X, f), axis = 1)
                X_names = np.concatenate((X_names, [f_i]))
            self.float_names = float_cols
        if isnull_cols != []:
            for n_i in isnull_cols:
                FG_i = FeatureGenerator(n_i + '_isnull')
                FG_i.feature_encoder(data[n_i].isnull())
                X = np.concatenate((X, FG_i.onehot_encoded), axis = 1)
                X_names = np.concatenate((X_names, [FG_i.feature_name + "_" + s for s in FG_i.class_labels]))            
        
        self.X = X
        self.X_names = X_names

    def product_matrix_preparer(self, lineitem_tape, top_pairs_num, categories_for_pairs = False, calculate_pairs = True):
        
        if all(i in list(lineitem_tape.columns) for i in ['order_id','customer_id','lineitem_id']) == False:
            sys.exit('column names are not standard')

        v = array(lineitem_tape['lineitem_id'])
        v = v.astype(str)

        label_encoder = LabelEncoder()
        label_encoder.fit(v)
        int_encoded_values = label_encoder.fit_transform(v)
        int_encoded_values = int_encoded_values.reshape(len(int_encoded_values),1)
    
        ohe = OneHotEncoder(sparse = False)
        ohev = ohe.fit_transform(int_encoded_values)

        lineitem_tape = pd.concat([lineitem_tape,
            pd.DataFrame(ohev, columns = ['lineitem_' + i for i in array(label_encoder.classes_).astype(str)])],
            axis = 1)

        order_product_matrix = lineitem_tape.drop(columns = ['customer_id','lineitem_id']).groupby('order_id', as_index=False).sum()

        customer_product_matrix = lineitem_tape.drop(columns = ['order_id','lineitem_id']).groupby('customer_id', as_index=False).sum()

        print('order & customer product matrices are ready')


        category = 'lineitem_id'

        if categories_for_pairs == True:
            category = 'lineitem_category'
            if not category in lineitem_tape.columns:
                sys.exit('category column is not standard')

        first_orders = lineitem_tape[lineitem_tape['order_id'].isin(
        		lineitem_tape.drop(columns = [category]).groupby('customer_id', as_index=False).min()['order_id'])]


        customer_first_product_matrix = first_orders.drop(columns=['order_id','lineitem_id']).groupby('customer_id',
            as_index = False).sum()

        new_names = ['customer_id']
        new_names.extend(['first_' + i for i in array(customer_first_product_matrix.columns).astype(str)][1:])

        customer_first_product_matrix.columns = new_names

        print('customer first product matrix is ready')

        if calculate_pairs:

            lineitem_tape['lineitem_pairs'] = ''
            for o_id in lineitem_tape['order_id'].unique():
                comb = [i for i in itertools.combinations(lineitem_tape[category][lineitem_tape['order_id'] == o_id], r =2)]
                lineitem_tape.loc[lineitem_tape['order_id'] == o_id, 'lineitem_pairs'] = pd.Series([comb]*len(lineitem_tape))

            all_pairs = []
            unique_tape = lineitem_tape.loc[lineitem_tape.groupby('order_id')['order_id'].idxmin()]
            for i in unique_tape['lineitem_pairs']:
                all_pairs.extend(list(i))

            pairs = [i[0] for i in Counter(all_pairs).most_common(top_pairs_num)]
            print('top ', top_pairs_num,' pairs: ', pairs)

            order_pairs = lineitem_tape[['order_id', 'customer_id', 'lineitem_id', 'lineitem_pairs']]

            for i in list(set(pairs)):
                order_pairs['lineitem_pair: ' + str(i)] = ''

            for j in range(order_pairs.shape[0]):
                p = [True in m for m in [[l == i for l in order_pairs['lineitem_pairs'].iloc[j]] for i in list(set(pairs))]]
                int_p = [int(x) for x in p]
                order_pairs.loc[j,['lineitem_pair: ' + str(n) for n in list(set(pairs))]] = int_p

            customer_order_pairs_matrix = order_pairs.drop(columns=['order_id','lineitem_id','lineitem_pairs']).groupby('customer_id',
                as_index = False).max()

            self.all_pairs = all_pairs
            self.customer_order_pairs_matrix = customer_order_pairs_matrix
        
        self.order_product_matrix = order_product_matrix
        self.customer_product_matrix = customer_product_matrix
        self.customer_first_product_matrix = customer_first_product_matrix
        
        