import pydot
import networkx as nx
from sklearn import tree
import graphviz
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


class SegmentsPreparer():
    def __init__(self, segments_name):
        self.segments_name = segments_name
        
    def find_leaves(self, X, clf):
        return set(clf.apply(X))
    
    def node_feature_values(self, X, clf, node=0, feature=0, require_leaf=False):

        leaf_ids = self.find_leaves(X, clf)
      
        if (require_leaf and
            node not in leaf_ids):
            print("<require_leaf> is set, "
                    "select one of these nodes:\n{}".format(leaf_ids))
            return

        node_indicator = clf.decision_path(X)
        node_array = node_indicator.toarray()

        samples_in_node_mask = node_array[:,node]==1

        return X[samples_in_node_mask, feature]
    
    def samples_in_node_mask(self, X, clf, node):
    
        leaf_ids = self.find_leaves(X, clf)

        node_indicator = clf.decision_path(X)
        node_array = node_indicator.toarray()

        samples_in_node_mask = node_array[:,node]==1

        return samples_in_node_mask

    def prepare_graph(self, X, feature_names, clf, label, feature=0, show_class_labels = False):
        graph = tree.export_graphviz(clf, feature_names = feature_names, out_file='tmp.dot', node_ids = True, proportion = True, filled = True,
                                 rounded = True)
        print(graph)
        dot_graph = pydot.graph_from_dot_file('tmp.dot')[0]


        MG = nx.nx_pydot.from_pydot(dot_graph)

        for n in range(clf.decision_path(X).toarray().shape[1]):
            nfv = self.node_feature_values(X, clf, node=n, feature=feature)
            MG.nodes[str(n)]['label'] = MG.nodes[str(n)]['label'] + "\\nnumber of records: {}".format(len(nfv))
            MG.nodes[str(n)]['label'] = MG.nodes[str(n)]['label'].replace("\"","")
 
            MG_str = np.array(str(MG.nodes[str(n)]['label']).replace("\"","").split('\\n'))
            MG_mask = ['gini' not in i for i in MG_str]
            MG.nodes[str(n)]['label'] = '\\n'.join(MG_str[MG_mask])
    
            if show_class_labels == False:
                MG.graph['graph'] = {'label':label, 'labelloc':"t"} 
            else:
                MG.graph['graph'] = {'label':label + '\\nclass labels: ' + str(clf.classes_), 'labelloc':"t"} 
            
            new_dot_data = nx.nx_pydot.to_pydot(MG)
        
        self.plot_data = new_dot_data
        
    def best_worst_feature_values(self, X, data, clf, cat_cols, float_cols, tree_type):
        if tree_type == 'classification':
            clf.classes_ = np.array([0,1])
        if tree_type == 'regression':
        	node_data = pd.DataFrame({'node': clf.apply(X), 'value': clf.predict(X)})
        
        elif tree_type == 'classification':
        	node_data = pd.DataFrame({'node': clf.apply(X),
                          'value': clf.predict_proba(X)[:,clf.classes_.argmax()]})

        else:
        	return('Unknown tree type')


        data[cat_cols] = data[cat_cols].astype(str)

        best_node = node_data['node'][node_data['value'].idxmax]
        worst_node = node_data['node'][node_data['value'].idxmin]
        
        self.best_node = best_node
        self.worst_node = worst_node
        
        self.best_cat_values = data[self.samples_in_node_mask(X, clf, best_node)][cat_cols].describe().transpose()
        self.worst_cat_values = data[self.samples_in_node_mask(X, clf, worst_node)][cat_cols].describe().transpose()
        
        self.best_float_values = data[self.samples_in_node_mask(X, clf, best_node)][float_cols].astype(float).describe().transpose()
        self.worst_float_values = data[self.samples_in_node_mask(X, clf, worst_node)][float_cols].astype(float).describe().transpose()