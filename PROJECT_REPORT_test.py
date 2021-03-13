# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:16:10 2021

@author: moham
"""

import unittest
import mock
import builtins

from PROJECT_REPORT import read_data
from PROJECT_REPORT import clean_data


class FileUtilsTest(unittest.TestCase):
    
    def setUp(self):
        self.expected = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME',
       'TIMESTAMP']
        self.expected_1 = {'CATEGORY': {'e': 17943, 'b': 13648, 't': 12976, 'm': 5433}}
    
    def test_read_file(self):
        self.assertCountEqual(read_data('uci-news-aggregator.csv').columns, self.expected)
        
    def test_clean_file(self):
        self.assertDictEqual(clean_data('uci-news-aggregator.csv').to_dict(), self.expected_1)