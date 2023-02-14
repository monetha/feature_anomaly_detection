import numpy as np

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, KMeans
from sklearn.decomposition import TruncatedSVD

import collections

import re
import string

class ExtraFeatureGenerator:
    def __init__(self, levels=2):
        self.features = {'size': set(['xxxs', '3xs', 'xxs', '2xs', 'xs', 's', 
                                  'm', 'l', 'xl', '2xl', 'xxl', '3xl', 'xxl']),
                         'volume': set(['ml', 'l']),
                         'weight': set(['g', 'kg', 'mg']),
                         'capacity': set(['mah']),
                         'power': set(['w']),
                         'length': set(['mm', 'cm', 'km']),
                         'frequency': set(['mhz', 'ghz', 'hz']),
                         'quantity': set(['pcs'])}
        self.levels = levels
        self.stop_words = []
        for key, value in self.features.items():
            self.stop_words += value
        self.stop_words = set(self.stop_words)
        
    def PrepareData(self, data):
        for level in range(self.levels):
            data.loc[:, 'label_' + str(level)] = np.NaN
        for key in self.features.keys():
            data.loc[:, key] = np.NaN
            
    def GetOptimalClusterNum(self, dist, N):
        n_clusters = 0
        prev_delta = dist[1] - dist[0]
        for i in range(1, len(dist) - 1):
            cur_delta = dist[i + 1] - dist[i]
            if cur_delta > N * prev_delta:
                n_clusters = i
                break
            prev_delta = cur_delta
        return len(dist) - n_clusters
    
    def Vectorize(self, names):
        words = set()
        for name in names:
            tokens = name.split()
            for token in tokens:
                words.add(token)
        word_to_idx = {}
        for i, word in enumerate(words):
            word_to_idx[word] = i
            
        matrix = np.zeros((len(names), len(words)))
        for i, name in enumerate(names):
            tokens = name.split()
            for token in tokens:
                matrix[i][word_to_idx[token]] += 1
        return matrix, word_to_idx
    
    def MakeOptimalClustering(self, data, N=2):
        matrix, _ = self.Vectorize(data['name'].tolist())
        for i in range(matrix.shape[0]):
            matrix[i] *= (hash(data.iloc[i]['name']) % int(1e9 + 7))
        matrix.astype(int)
        n_clusters = matrix.shape[0] - 1
        
        if matrix.shape[0] > 1e4: 
            svd = TruncatedSVD(n_components=200, random_state=123)
            matrix = svd.fit_transform(matrix)
            
        clusterizer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', 
                                              compute_distances=True)
        
        if matrix.shape[1] < 1:
            labeled_data = data.copy()
            labeled_data['label'] = np.NaN
            return labeled_data
        
        labels = clusterizer.fit_predict(matrix)
        if len(clusterizer.distances_) <= 1:
            n_clusters = 1
        else:
            n_clusters = max(self.GetOptimalClusterNum(clusterizer.distances_, N), 1)
            
        if matrix.shape[0] > 1e4:
            opt_clu = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            opt_labels = opt_clu.fit_predict(matrix)
        else:
            opt_clu = KMeans(n_clusters=n_clusters, random_state=42)
            opt_labels = opt_clu.fit_predict(matrix)
        
        labeled_data = data.copy()
        labeled_data['label'] = opt_labels
        return labeled_data
    
    def CheckIsWordOk(self, name):
        words = str(name).split('-')
        for word in words:
            if not re.match("^[A-Za-z]*$", word):
                return False
        if str(word).lower() in self.stop_words:
            return False
        
        return len(str(word)) > 0 and str(word)[0].isupper()

    
    def FindTopWords(self, data):
        X, word_to_idx = self.Vectorize(data['name'].tolist())
        number_of_appearances = {}
        for word in word_to_idx.keys():
            number_of_appearances[word] = np.sum(X.T[word_to_idx[word]])
        max_value = 0
        best_words = []
        stop_words = set(['the', 'ml', 'vnt.', 'co.'])
        for key, value in number_of_appearances.items():
            if value > max_value and self.CheckIsWordOk(key):
                best_words = [key]
                max_value = value
            elif value == max_value and self.CheckIsWordOk(key):
                best_words.append(key)
        if len(best_words) >= 1:
            return best_words, best_words[0]
        else:
            return -1, np.NaN
        
    
    def FixName(self, name, top_words):
        new_name = name.split()
        try:
            s_t = set(top_words)
            return " ".join([word for word in new_name if word not in s_t]).strip()
        except TypeError:
            return name
        
    
    def RemoveWord(self, data):
        new_cluster = data.copy()
        top_words, _ = self.FindTopWords(new_cluster)
        names = set(new_cluster['name'].tolist())
        for name in names:
            new_cluster.loc[new_cluster['name'] == name, 'name'] = self.FixName(name, top_words)
        return new_cluster
    
    def GenerateLabels(self, data, levels=2):
        #print(levels)
        num_iterations = 0
        data_queue = collections.deque([])
        data_queue.append([data.copy(), 0])
        if data.shape[0] > 1e4:
            levels = min(2, levels)
        while len(data_queue) > 0:
            cur_list = data_queue.popleft()
            
            num_iterations += 1
            
            cur_data, cur_level = cur_list[0], cur_list[1]
            new_data = self.RemoveWord(cur_data)
            
            if new_data.shape[0] < 2:
                for i in range(cur_level + 1, levels):
                    del data['label_' + str(i)]
                break
                
            lab_data = self.MakeOptimalClustering(new_data, N=1.2)
        
            for l in set(lab_data['label']):
                _, top_word = self.FindTopWords(lab_data[lab_data['label'] == l])
                lab_data.loc[lab_data['label'] == l, 'label'] = top_word
            
            if np.sum(lab_data['label'].isna()) > 0.1 * lab_data['label'].shape[0]:
                continue
            
            cur_label = 'label_' + str(cur_level)
            
            cur_data[cur_label] = lab_data['label']
            cur_data['name'] = lab_data['name']
            indices = cur_data.index.tolist()
            data.loc[indices, cur_label] = cur_data[cur_label]
            #print(cur_data.loc[:, 'name':].head(2))
            
            
            if cur_level < levels:
                cur_level += 1
                labels = set(cur_data[cur_label].tolist())
                for label in labels:
                    data_queue.append([cur_data[cur_data[cur_label] == label].copy(), cur_level]) 
        
        return data, num_iterations
    
    
    def GenerateExtraFeatures(self, data):
        percentage = dict.fromkeys(self.features.keys(), 0)
        nominatives = {}
        for key in self.features.keys():
            nominatives[key] = []
            percentage[key] = 0
            
        names = data['name']
        for name in names:
            bigrams = self.GetBigrams([name])
            s_bi = set([b[1] for b in bigrams])
            for key in self.features.keys():
                if len(self.features[key].intersection(s_bi)) >= 1:
                    for bi in bigrams:
                        if bi[1] in self.features[key]:
                            nominatives[key].append(str(bi[0]) + ' ' + str(bi[1]))
                            percentage[key] += 1
                            break
                else:
                    nominatives[key].append(np.NaN)
        succeeded_cols = []
        for key in percentage.keys():
            if percentage[key] > 0.05 * data['name'].shape[0]:
                succeeded_cols.append(key)
        new_data = data.copy()
        for col in succeeded_cols:
            new_data[col] = nominatives[col]
        return new_data
                    
    
    def GenerateFeatures(self, data):
        num_pops = 0
        self.PrepareData(data)
        cur_data = self.GenerateExtraFeatures(data)
        cur_data, cur_pops = self.GenerateLabels(cur_data, levels=self.levels)
        num_pops += cur_pops
        return cur_data, num_pops
    
    def SplitDigitPlusStr(self, word):
        for i in range(1, len(word)):
            l, r = word[:i], word[i:]
            if l.isdigit() and re.match("^[A-Za-z]*$", r):
                return l, r
        return -1, -1
    
    def GetBigramsFromData(self, data):
        names = data['name'].tolist()
        return self.GetBigrams(names)
        
    def GetBigrams(self, names):
        bigrams = set()
        s_part = set()
        for name in names:
            banned = '!@#$,\"/|\'.:'
            for char in banned:
                name = name.replace(char, ' ')
            words = name.strip().split()
            correct_words = []
            for word in words:
                l, r = self.SplitDigitPlusStr(word)
                if l != -1:
                    correct_words += [l, r]
                else:
                    correct_words.append(word)
            for i in range(1, len(correct_words)):
                if correct_words[i - 1].isdigit() and not correct_words[i].isdigit():
                    bigrams.add((correct_words[i - 1], correct_words[i]))
                    s_part.add(correct_words[i])
        return list(bigrams)
        
