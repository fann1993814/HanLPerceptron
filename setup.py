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

from os.path import abspath, join, dirname
import sys
from setuptools import find_packages, setup

this_dir = abspath(dirname(__file__))
if sys.version_info[0] < 3:  # In Python3 TypeError: a bytes-like object is required, not 'str'
    long_description = 'Native Python HanLP Perceptron Model: HanLPerceptron'
else:
    with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
        long_description = file.read()

setup(
    name='hanlperceptron',
    version='0.2.0',
    description='Native Python HanLP Perceptron Model: HanLPerceptron',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/fann1993814/HanLPerceptron',
    author='Jason Fan',
    author_email='fann1993814@gmail.com',
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: Chinese (Traditional)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic'
    ],
    keywords='corpus,machine-learning,NLU,NLP',
    python_requires='>=3.6',
    packages=find_packages(exclude=['tests*']),
    package_dir={'hanlperceptron':'hanlperceptron'},
    package_data={'':['*.*','data/*']},
    include_package_data=True,
    install_requires=['numpy>=1.15','joblib>=0.14.0']
)