from toxicity.data import load 
import numpy as np
import pandas as pd 
import unittest

class TestLoad(unittest.TestCase):
    def setUp(self):
        self.data_file_path = "dummy_data/dummy.txt"

    def test_default_load(self):
        df = load(self.data_file_path)
        types = [np.dtype('O'), np.dtype('int64')]
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df.columns), 2)
        self.assertTrue(np.all(df.dtypes==types))

    def test_filter_false(self):
        df = load(self.data_file_path, filter=False)
        types = 2*[np.dtype('O')] + 6*[np.dtype('int64')]
        self.assertEqual(len(df), 6)
        self.assertEqual(len(df.columns), 8)
        self.assertTrue(np.all(df.dtypes==types))

    def test_filter_true_returnXy_true(self):
        X,y = load(self.data_file_path, filter=True, return_X_y=True)
        types = [np.dtype('O')]
        self.assertEqual(len(X), 6)
        self.assertEqual(len(y), 6)
        self.assertEqual(len(X.columns), 1)
        self.assertTrue(np.all(X.dtypes==types))

    def test_filter_false_returnXy_true(self):
        with self.assertRaises(ValueError):
            load(self.data_file_path, filter=False, return_X_y=True)
        