# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 18:16:10 2021

@author: moham
"""

import unittest

from PROJECT_REPORT import read_data
from PROJECT_REPORT import clean_data


class FileUtilsTest(unittest.TestCase):
    """Class for testing file operations."""
    def setUp(self):
        """Sets up expected output."""
        self.expected = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME',
       'TIMESTAMP']
        self.expected_1 = {'CATEGORY': {'e': 17943, 'b': 13648, 't': 12976, 'm': 5433}}
    def test_read_file(self):
        """Test reading the correct file."""
        self.assertCountEqual(read_data('uci-news-aggregator.csv').columns, self.expected)
    def test_clean_file(self):
        """Test post data processing and file cleaning."""
        self.assertDictEqual(clean_data('uci-news-aggregator.csv').to_dict(), self.expected_1)
