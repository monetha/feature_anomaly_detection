import pandas as pd
import numpy as np
import datetime as dt
import time
def save_graph(_SP,path,name):
    _SP.plot_data.write_png(f'{path}/{name}_graph.png')

class StatSaver():
    def __init__(self, markup_time) -> None:
        self.markup_list = []
        self.rules_list = []
        self.markup_time = markup_time
        self.attr_method = {
            'regression' : self.__predict,
            'classification' : self.__predict_proba
        }
        
    def save(self, path, model_type):
        pd.concat(self.markup_list).to_csv(path + f'/markup_{model_type}.csv')
        pd.concat(self.rules_list).to_csv(path + f'/rules_{model_type}.csv')

    def __predict_proba(self, tree, X):
        return [i[1] for i in tree.predict_proba(X)]

    def __predict(self, tree, X):
        return tree.predict(X)
    def save_stats(self, SP, FG, train_data, tree, cat_features, float_features, path, name, customer_id, ACCOUNT_ID, tree_type):
        SP.best_worst_feature_values(FG.X, train_data, tree,
                                        cat_cols=cat_features, float_cols=float_features, tree_type=tree_type)

        spr_markup = pd.DataFrame({customer_id: train_data[customer_id],
                                'predicted_value': self.attr_method[tree_type](tree,FG.X)})
        spr_markup['segment'] = ''
        segments = SP.find_leaves(FG.X, tree)
        for i in segments:
            idx_segment = np.where(SP.samples_in_node_mask(FG.X, tree, i) == True)[0]

            spr_markup['segment'].iloc[idx_segment] = i

        spr_markup['model'] = name
        spr_markup['markup_datetime'] = self.markup_time
        spr_markup['account_id'] = ACCOUNT_ID

        self.markup_list.append(spr_markup)

        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        def find_path(node_numb, path, x):
            path.append(node_numb)
            if node_numb == x:
                return True
            left = False
            right = False
            if (children_left[node_numb] != -1):
                left = find_path(children_left[node_numb], path, x)
            if (children_right[node_numb] != -1):
                right = find_path(children_right[node_numb], path, x)
            if left or right:
                return True
            path.remove(node_numb)
            return False
        def get_rule(path, column_names):
            mask = ''
            for index, node in enumerate(path):
                # We check if we are not in the leaf
                if index != len(path) - 1:
                    # Do we go under or over the threshold ?
                    if (children_left[node] == path[index + 1]):
                        mask += "('{}' DOES NOT EXIST) \t ".format(column_names[feature[node]])
                    else:
                        mask += "('{}' EXISTS) \t ".format(column_names[feature[node]])
            # We insert the & at the right places
            #    mask = mask.replace("\t", "&\n", mask.count("\t") - 1)
            mask = mask.replace("\t", "& <br>", mask.count("\t") - 1)
            mask = mask.replace("\t", "")
            return mask
        # Leaves
        leave_id = tree.apply(FG.X)

        paths ={}
        for leaf in np.unique(leave_id):
            path_leaf = []
            find_path(0, path_leaf, leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))

        rules = {}
        for key in paths:
            rules[key] = get_rule(paths[key], FG.X_names)

        spr_rules_df = pd.DataFrame.from_dict(rules, orient='index', columns=['description']).reset_index()
        spr_rules_df = spr_rules_df.rename(columns = {'index': 'segment'})
        spr_rules_df['model'] = name
        spr_rules_df['markup_datetime'] = self.markup_time
        spr_rules_df['account_id'] = ACCOUNT_ID
        self.rules_list.append(spr_rules_df)
        return spr_markup, spr_rules_df
        # spr_markup.to_csv(f'{path}/markup_{name}.csv')
        # spr_rules_df.to_csv(f'{path}/rules_{name}.csv')