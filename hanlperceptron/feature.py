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

import numpy as np
from hanlperceptron.utils import ByteArray

class Table:
    def __init__(self, ):
        self.size = 0
        self.data = []
        self.linearExpandFactor = 0
        self.exponentialExpanding = False
        self.exponentialExpandFactor =0.
    
    def load(self, bytearray):
        self.size = bytearray.nextInt()
        self.data = np.array([bytearray.nextInt() for i in range(self.size)], dtype=np.int64)
        self.linearExpandFactor = bytearray.nextInt()
        self.exponentialExpanding = bytearray.nextBoolean()
        self.exponentialExpandFactor = bytearray.nextDouble()
    
    def get(self, idx):
        return self.data[idx]
    
    def get_size(self,):
        return self.size

class FeatureMap:
    def __init__(self, ):
        self.base = Table()
        self.check = Table()
        self.UNUSED_CHAR_VALUE = ord('\000')
        self.LEAF_BIT = 1073741824
        
    def load(self, bytearray):
        self.base.load(bytearray)
        self.check.load(bytearray)
    
    def idOf(self, string):
        state = 1;
        ids = self._toIdList(string)
        state = self._transfer(state, ids)
        if state < 0:
            return -1
        return self._stateValue(state)
    
    def _toIdList(self, string):
        _bytes = string.encode('utf-8')
        res = [_bytes[i] & 0xFF for i in range(0, len(_bytes))]
        if (len(res) == 1) and (res[0] == 0):
            return [0]
        return res
        
    def _transfer(self, state, ids):
        for c in ids:
            if ((self.base.get(state) + c < self.base.get_size()) and \
                (self.check.get(self.base.get(state) + c) == state)):
                state = self.base.get(state) + c
            else:
                return -1
        return state
    
    def _stateValue(self, state):
        leaf = self.base.get(state) + self.UNUSED_CHAR_VALUE
        if self.check.get(leaf) == state:
            return self.base.get(leaf) ^ self.LEAF_BIT
        return -1;
