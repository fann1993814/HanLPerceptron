# -*- coding:utf-8 -*-
# Author：Jason
# Reference: HanLP Project v1.x https://github.com/hankcs/HanLP
# Note: The Project refers to HanLP, so it follows the HanLP's License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import joblib
from hanlperceptron.model import LinearModel
from hanlperceptron.instance import CWSInstance, POSInstance, NERInstance
from hanlperceptron.other import CharTable

word2POS = {}
char_table = CharTable()

class DictAnalyzer:
    def __init__(self, ):
        self.FREQ = {}
        self.total = 0
        
    def init(self, file):
        global word2POS
        
        with open(file, 'r', encoding='utf-8') as fr:
            for line in fr:
                field = line.strip().split(' ')
                
                if len(field) == 1:
                    word = field[0]
                    self.FREQ[word] = 1
                    
                elif len(field) == 2:
                    word, freq = field
                    self.FREQ[word] = int(freq)
                    
                elif len(field) == 3:
                    word, freq, pos = field
                    self.FREQ[word] = int(freq)
                    word2POS[word] = pos
                
                for ch in range(len(word)):
                    wfrag = word[:ch + 1]
                    if wfrag not in self.FREQ:
                        self.FREQ[wfrag] = 0
                        
                self.total += self.FREQ[word]
        
    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)
        logtotal = math.log(self.total)
        for idx in range(N - 1, -1, -1):
            route[idx] = max((math.log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                              logtotal + route[x + 1][0], x) for x in DAG[idx])
        
    def get_DAG(self, sentence):
        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N and frag in self.FREQ:
                if self.FREQ[frag]:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG
    
    def segment(self, sentence, model_func=None):
        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)
        x = 0
        N = len(sentence)
        buf = ''
        result = []
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if len(l_word) == 1:
                buf += l_word
            else:
                if buf:
                    if model_func: result += model_func(buf)
                    else: result.append(buf)
                    buf = ''
                result.append(l_word)
            x = y
            
        if buf:
            if model_func: result += model_func(buf)
            else: result.append(buf)
            buf = ''
            
        return result

class Segmenter:
    def __init__(self, file=None, custom_dict=None):
        self.model = LinearModel()
        self.enable_dict = False
        self.dictanalyzer = DictAnalyzer()
        
        if file:
            self.load(file)
            
        if custom_dict:
            self.load_custom_dict(custom_dict)
    
    def segment(self, sentence, viterbi=True):
    
        if len(sentence) == 1:
            return [sentence]
        
        normalize_sent = char_table.normalize(sentence)
        instance = CWSInstance(normalize_sent, self.model.feature_map)
        
        if viterbi:
            result = self.model.viterbiDecode(instance)
        else:
            result = self.model.greedyDecode(instance)
        
        last = 0
        w_list = []
        seq_len = len(result)
        
        for i in range(seq_len):
            if result[i] in 'ES':
                w_list.append(sentence[last:i+1])
                last = i+1
                
        if last != seq_len:
            w_list.append(sentence[last:])
        
        return w_list
    
    def segment_with_dict(self, sentence):
        if self.enable_dict:
            return self.dictanalyzer.segment(sentence, self.segment)
        else:
            return self.segment(sentence)

    def save(self, file):
        joblib.dump(self.model, file, compress=3)
        
    def load(self, file):
        load_from_pkl_fail = False
        
        try:
            self.model = joblib.load(file)
        except Exception:
            load_from_pkl_fail = True
        
        if load_from_pkl_fail:
            self.model.load(file)
        
    def load_custom_dict(self, file):
        self.enable_dict = True
        self.dictanalyzer.init(file)
        
class POSTagger:
    def __init__(self, file=None):
        self.model = LinearModel()
        
        if file:
            self.load(file)
    
    def tag(self, w_list, viterbi=True):
        normalize_w_list = [char_table.normalize(word) for word in w_list]
        instance = POSInstance(normalize_w_list, self.model.feature_map)
        
        if viterbi:
            result = self.model.viterbiDecode(instance)
        else:
            result = self.model.greedyDecode(instance)
        
        return result
    
    def tag_with_dict(self, w_list, viterbi=True):
        result = self.tag(w_list, viterbi)
        result = [word2POS[w] if w in word2POS else result[i] for i, w in enumerate(w_list)]
        return result

    def save(self, file):
        joblib.dump(self.model, file, compress=3)
        
    def load(self, file):
        load_from_pkl_fail = False
        
        try:
            self.model = joblib.load(file)
        except Exception:
            load_from_pkl_fail = True
        
        if load_from_pkl_fail:
            self.model.load(file)
            
class NERecognizer:
    def __init__(self, file=None):
        self.model = LinearModel()
        
        if file:
            self.load(file)
    
    def recognize(self, w_list, p_list, viterbi=True):
        normalize_w_list = [char_table.normalize(word) for word in w_list]
        instance = NERInstance(normalize_w_list, p_list, self.model.feature_map)
        
        if viterbi:
            result = self.model.viterbiDecode(instance)
        else:
            result = self.model.greedyDecode(instance)
        
        return result
    
    def save(self, file):
        joblib.dump(self.model, file, compress=3)
        
    def load(self, file):
        load_from_pkl_fail = False
        
        try:
            self.model = joblib.load(file)
        except Exception:
            load_from_pkl_fail = True
        
        if load_from_pkl_fail:
            self.model.load(file)