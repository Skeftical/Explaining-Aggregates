import unittest
import os
os.chdir("../../explanation_framework")
import sys
sys.path.append('codebase')
from framework.Preprocessing import PreProcessing as PR
import numpy as np
from sklearn.linear_model import LinearRegression

class PreProcessingTestCase(unittest.TestCase):
    def setUp(self):
        lr = LinearRegression()
        self.pr = PR(lr)
        self.ex = np.random.uniform(size=(100,4))
        self.pr.fit(self.ex, np.random.uniform(size=100))

    def test_l1_cluster_dimensions(self):
        cc = self.pr.get_cluster_centers()
        self.assertEqual(cc.shape[1], 2,
                         'incorrect cluster dimensions')
    def test_l2_cluster_dimensions(self):
        cc = self.pr.get_theta_centers()
        K = self.pr.get_cluster_centers().shape[0]
        self.assertEqual(len(cc), K, "Incorrect Number of L2 clusters")
        self.assertIsInstance(cc, dict, "Theta Centers not a dictionary instance")
        for k in cc:
            self.assertEqual(cc[k].shape[1], 2,
                         'incorrect theta cluster dimensions')

    def test_models(self):
        m = self.pr.get_models()
        L = self.pr.get_theta_centers()
        tot = 0
        for u in L:
            tot+=L[u].shape[0]
        self.assertEqual(len(m), tot, "Incorrect number of models")

    def test_closest_l1(self):
        mock_centers = np.random.uniform(low=20,high=30,size=(5,2))
        mock_centers = np.row_stack((mock_centers,np.array([0,0])))
        mock_true_center = 5
        self.pr.CLUSTER_CENTERS = mock_centers
        q = np.array([0 ,0 ,0 ,0]).reshape(1,4)
        self.assertEqual(self.pr.get_closest_l1(q), 5 ,
                                        "Closest L1 representantive incorrect")

    def test_closest_l2(self):
        mock_centers = np.random.uniform(low=20,high=30,size=(5,2))
        mock_centers = np.row_stack((mock_centers,np.array([0,0])))
        mock_true_center = 5
        self.pr.CLUSTER_CENTERS = mock_centers
        self.pr.THETA_CENTERS[5] = mock_centers
        q = np.array([0 ,0 ,0 ,0]).reshape(1,4)
        self.assertEqual(self.pr.get_closest_l2(q), (5,5) ,
                                        "Closest L2 representantive incorrect")
if __name__ == '__main__':
    unittest.main()
