from toxicity.model import apply_count_vectorizer, run_naive_bayes_classifier
import numpy as np
import pandas as pd 
import unittest
from joblib import dump, load
import os
from sklearn.naive_bayes import MultinomialNB


class TestApplyCountVectorizer(unittest.TestCase):
    def setUp(self):
        data_file_path = "dummy_data/xdummy.txt"
        with open(data_file_path) as f:
            self.xdummy = f.readlines()

    def test_count_vectorizer(self):
        count_train, count_valid = apply_count_vectorizer(self.xdummy, self.xdummy)
        xmatrix = [[1, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0]]
        self.assertEqual(count_train.shape, (7,4))
        self.assertTrue(np.all(count_train.todense()==count_train.todense()))
        self.assertTrue(np.all(xmatrix==count_train.todense()))

    def test_output_path(self):
        output_path = "dummy_data"
        count_train, count_valid = apply_count_vectorizer(
            self.xdummy, 
            self.xdummy, 
            output_path=output_path)
        self.assertTrue(os.path.exists(output_path + "/count_vectorizer.joblib"))
        os.remove(output_path + "/count_vectorizer.joblib")


class TestRunNaiveBayes(unittest.TestCase):
    def setUp(self):
        self.count_train = [[1, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 1, 0]]
        self.ytrain = [1, 1, 1, 0, 0, 0, 1]

    def test_instance(self):
        self.assertIsInstance(run_naive_bayes_classifier(self.count_train, self.ytrain),
        MultinomialNB)

    def test_output_path(self):
        output_path = "dummy_data"
        nb_classifier = run_naive_bayes_classifier(
            self.count_train, 
            self.ytrain, 
            output_path=output_path)
        self.assertTrue(os.path.exists(output_path + "/nb_classifier.joblib"))
        os.remove(output_path + "/nb_classifier.joblib")
