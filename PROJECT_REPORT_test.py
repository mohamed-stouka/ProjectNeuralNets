# -*- coding: utf-8 -*-


import unittest
from read_data import read_data
from clean_data import clean_data

cat_count, dt = clean_data('uci-news-aggregator.csv')
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
        self.assertDictEqual(cat_count.to_dict(), self.expected_1)
