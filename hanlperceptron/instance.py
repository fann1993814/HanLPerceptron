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

from hanlperceptron.feature import FeatureMap

class CWSInstance:
    def __init__(self, sequence, featureMap):
        self.CHAR_BEGIN = '\u0001'
        self.CHAR_END = '\u0002'
        
        self.featureMatrix = [self.extractFeature(sequence, featureMap, i) \
                              for i in range(len(sequence))]
        
    def extractFeature(self, sequence, featureMap, position):
        
        featureVector = []
        
        pre2Char = sequence[position - 2] if position >= 2 else self.CHAR_BEGIN
        preChar = sequence[position - 1] if position >= 1 else self.CHAR_BEGIN
        curChar = sequence[position]
        nextChar = sequence[position + 1] if position <= len(sequence) - 2 else self.CHAR_END
        next2Char = sequence[position + 2] if position <= len(sequence) - 3 else self.CHAR_END
        
        #char unigram feature
        self.addFeature(preChar + '1', featureVector, featureMap)
        self.addFeature(curChar + '2', featureVector, featureMap)
        self.addFeature(nextChar + '3', featureVector, featureMap)
        
        #char bigram feature
        self.addFeature(pre2Char + '/' + preChar + '4', featureVector, featureMap)
        self.addFeature(preChar + '/' + curChar + '5', featureVector, featureMap)
        self.addFeature(curChar + '/' + nextChar + '6', featureVector, featureMap)
        self.addFeature(nextChar + '/' + next2Char + '7', featureVector, featureMap)
        
        #transitionFeature feature (last)
        featureVector.append(-1)
        
        return featureVector
        
    def addFeature(self, rawFeature, featureVector, featureMap):
        id = featureMap.idOf(rawFeature)
        if id != -1:
            featureVector.append(id)
    
    def getFeatureAt(self, index):
        return self.featureMatrix[index]

class POSInstance:
    def __init__(self, sequence, featureMap):
        
        self.featureMatrix = [self.extractFeature(sequence, featureMap, i) \
                              for i in range(len(sequence))]
        
    def extractFeature(self, sequence, featureMap, position):
        
        featureVector = []
        
        preWord = sequence[position - 1] if position >= 1 else "_B_"
        curWord = sequence[position]
        nextWord = sequence[position + 1] if position <= len(sequence) - 2 else "_E_"
        
        #word unigram feature
        self.addFeature(preWord + '1', featureVector, featureMap)
        self.addFeature(curWord + '2', featureVector, featureMap)
        self.addFeature(nextWord + '3', featureVector, featureMap)
        
        #prefix and suffix feature
        length = len(curWord)
        self.addFeature(curWord[:1] + '4', featureVector, featureMap)
        self.addFeature(curWord[-1] + '5', featureVector, featureMap)
        if length > 1:
            self.addFeature(curWord[:2] + '4', featureVector, featureMap)
            self.addFeature(curWord[-2:] + '5', featureVector, featureMap)
        if length > 2:
            self.addFeature(curWord[:3] + '4', featureVector, featureMap)
            self.addFeature(curWord[-3:] + '5', featureVector, featureMap)
        
        #transitionFeature feature (last)
        featureVector.append(-1)
        
        return featureVector
        
    def addFeature(self, rawFeature, featureVector, featureMap):
        id = featureMap.idOf(rawFeature)
        if id != -1:
            featureVector.append(id)
            
    def getFeatureAt(self, index):
        return self.featureMatrix[index]

class NERInstance:
    def __init__(self, wordArray, posArray, featureMap):
        
        self.featureMatrix = [self.extractFeature(wordArray, posArray, featureMap, i) \
                              for i in range(len(wordArray))]
        
    def extractFeature(self, wordArray, posArray, featureMap, position):
        
        featureVector = []
        
        pre2Word = wordArray[position - 2] if position >= 2 else "_B_"
        preWord = wordArray[position - 1] if position >= 1 else "_B_"
        curWord = wordArray[position]
        nextWord = wordArray[position + 1] if position <= len(wordArray) - 2 else "_E_"
        next2Word  = wordArray[position + 2] if position <= len(wordArray) - 3 else "_E_"
        
        pre2Pos = posArray[position - 2] if position >= 2 else "_B_"
        prePos = posArray[position - 1] if position >= 1 else "_B_"
        curPos = posArray[position]
        nextPos = posArray[position + 1] if position <= len(posArray) - 2 else "_E_"
        next2Pos = posArray[position + 2] if position <= len(posArray) - 3 else "_E_"
        
        #word unigram feature
        self.addFeature(pre2Word + '1', featureVector, featureMap)
        self.addFeature(preWord + '2', featureVector, featureMap)
        self.addFeature(curWord + '3', featureVector, featureMap)
        self.addFeature(nextWord + '4', featureVector, featureMap)
        self.addFeature(next2Word + '5', featureVector, featureMap)
        
        #pos unigram feature
        self.addFeature(pre2Pos + 'A', featureVector, featureMap)
        self.addFeature(prePos + 'B', featureVector, featureMap)
        self.addFeature(curPos + 'C', featureVector, featureMap)
        self.addFeature(nextPos + 'D', featureVector, featureMap)
        self.addFeature(next2Pos + 'E', featureVector, featureMap)
        
        #pos bigram feature
        self.addFeature(pre2Pos + prePos + 'F', featureVector, featureMap)
        self.addFeature(prePos + curPos + 'G', featureVector, featureMap)
        self.addFeature(curPos + nextPos + 'H', featureVector, featureMap)
        self.addFeature(nextPos + next2Pos + 'I', featureVector, featureMap)
        
        #transitionFeature feature (last)
        featureVector.append(-1)
        
        return featureVector
        
    def addFeature(self, rawFeature, featureVector, featureMap):
        id = featureMap.idOf(rawFeature)
        if id != -1:
            featureVector.append(id)
            
    def getFeatureAt(self, index):
        return self.featureMatrix[index]