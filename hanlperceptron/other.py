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

import os

get_module_path = lambda *res: os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), *res))

try:
    import pkg_resources
    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__), *res)), 'rb')

class CharTable:
    def __init__(self, file='data/CharTable.txt'):
        self.table = {}
        with get_module_res(file) as fr:
            for line in fr:
                line = line.decode('utf-8')
                field = line.strip().split('=')
                if len(field) == 2:
                    self.table[field[0]] = field[1]
        
    def normalize(self, sentence):
        sentence = ''.join([self.table[ch] if self.table.get(ch) else ch for ch in sentence])
        return sentence