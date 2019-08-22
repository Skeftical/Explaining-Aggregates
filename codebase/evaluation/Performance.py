import pandas as pd
import numpy as np
import os
import sys
import fnmatch

os.chdir("../../../explanation_framework")
sys.path.append("../explanation_framework")
sys.path.append('utils')
from terminal_outputs import printProgressBar
from confs import Config
from setup import logger
#Models
from pyearth import Earth
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.stats import entropy
from codebase.framework.Preprocessing import PreProcessing as PR
import argparse
import logging
import time
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument('-v',help='verbosity',dest='verbosity',action="store_true")

args = parser.parse_args()
if args.verbosity:
   print("verbosity turned on")
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(logging.DEBUG)
   logger.addHandler(handler)
if not os.path.exists('output/Performance'):
        logger.info('creating directory Performance')
        os.makedirs('output/Performance')

def execution_time(train_df):
        X_train = train_df[['x','y','x_range','y_range']].values
        y_train = train_df['count'].values
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        #Training Models
        logger.info("Model Training Initiation\n=====================")
        kmeans = KMeans(random_state=0)
        mars_ = Earth(feature_importance_type='gcv',)

        lsnr = PR(mars_)
        start = time.time()
        lsnr.fit(X_train,y_train)
        return (time.time()-start, lsnr.get_number_of_l1(), lsnr.get_number_of_l2())

if __name__=='__main__':
    np.random.seed(15)
    logger.info("Finding datasets...")
    train_df = pd.read_csv('/home/fotis/dev_projects/explanation_framework/input/Crimes_Workload/train_workload_x-gauss-length-gauss-5-users-50000.csv', index_col=0)
    initial = train_df.head(10000)
    part = train_df.head(5000)
    data = {'size' : [], 'time':[], 'l1':[],'l2':[]}
    for i in range(10):
        initial = pd.concat([initial, part])
        for j in range(10):
            t,l1,l2 = execution_time(part)
            data['size'].append(initial.count()[0])
            data['time'] .append(t)
            data['l1'].append(l1)
            data['l2'].append(l2)
            logger.info("Loop {}/100".format(j+10**i))
    eval_df = pd.DataFrame(data)
    eval_df.to_csv('output/Performance/evaluation_results_training_time.csv')
