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

import mmap
import numpy as np

from hanlperceptron.utils import ByteArray
from hanlperceptron.feature import FeatureMap

class LinearModel:
    def __init__(self, ):
        self.task = ''
        self.class_size = 0
        self.id2tag = {}
        self.feature_size = -1
        self.feature_map = FeatureMap()
        self.parameter = None
    
    def load(self, file):
        
        with open(file, 'r+b') as fr:
            buffer = mmap.mmap(fr.fileno(), 0)
        
        bytearray = ByteArray(buffer)
            
        task_types = ['cws', 'pos', 'ner']
        
        self.task = task_types[bytearray.nextInt()]
        self.class_size = bytearray.nextInt()
        self.id2tag = {i:bytearray.nextUTF() for i in range(self.class_size)}
        
        self.feature_size = bytearray.nextInt()
        self.feature_map.load(bytearray)
        
        self.parameter = np.array([bytearray.nextFloat() \
                                   for i in range(0, self.feature_size * self.class_size)], dtype=np.float32)
    
    def greedyDecode(self, instance):
        
        bos = self.class_size # for first transition fid
        sentenceLength = len(instance.featureMatrix)
        
        prev = bos
        guessLabel = ['' for i in range(sentenceLength)]
        
        for i in range(0, sentenceLength):
            featureVector = instance.getFeatureAt(i)
            scores = [self._score(featureVector, j, prev) for j in range(0, self.class_size)]
            maxIndex, _ = self._max(scores)
            prev = maxIndex
            guessLabel[i] = self.id2tag[maxIndex]
        
        return guessLabel
        
    def viterbiDecode(self, instance):
        
        bos = self.class_size # for first transition fid
        sentenceLength = len(instance.featureMatrix)
        
        guessLabel = ['' for i in range(sentenceLength)]
        preMatrix = [[-1 for j in range(self.class_size)] for i in range(sentenceLength)]
        scoreMatrix = [[0.0 for j in range(self.class_size)] for i in range(2)]
        
        for i in range(0, sentenceLength):
            _i = i & 1
            _i_1 = 1 - _i
            
            featureVector = instance.getFeatureAt(i)
            
            if 0 == i:
                for curLabel in range(self.class_size):
                    preMatrix[0][curLabel] = curLabel
                    score = self._score(featureVector, curLabel, bos)
                    scoreMatrix[0][curLabel] = score
            else:
                for curLabel in range(0, self.class_size):
                    cache = self._score(featureVector, curLabel)
                    scores = [scoreMatrix[_i_1][preLabel] + self._score(featureVector, curLabel, preLabel, cache) \
                                         for preLabel in range(0, self.class_size)]
                    maxIndex, maxScore = self._max(scores)
                    preMatrix[i][curLabel] = maxIndex
                    scoreMatrix[_i][curLabel] = maxScore
                    
        maxIndex, maxScore = self._max(scoreMatrix[(sentenceLength - 1) & 1])
        
        for i in range(sentenceLength - 1, -1, -1):
            guessLabel[i] = self.id2tag[maxIndex]
            maxIndex = preMatrix[i][maxIndex]
            
        return guessLabel
    
    def _max(self, l):
        max_val = max(l)
        max_idx = l.index(max_val)
        return (max_idx, max_val)
    
    def _score(self, featureVector, currentTag, prevTag=None, cache=None):
        if prevTag is not None:
            if cache is not None:
                score = cache + self.parameter[prevTag * self.class_size + currentTag]
            else:
                featureVector[-1] = prevTag
                score = sum([self.parameter[index * self.class_size + currentTag] \
                             for index in featureVector \
                             if index != -1])
        else:
            score = sum([self.parameter[index * self.class_size + currentTag] \
                         for index in featureVector[:-1] \
                         if index != -1])
                         
        return score
    