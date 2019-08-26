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

def training_time(train_df):
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
    return data

def explanation_serving_x(train_df):
    data = {'vigil_x' : [], 'explanation_serving_time':[], 'l1':[],'l2':[]}
    X_train = train_df[['x','y','x_range','y_range']].values
    y_train = train_df['count'].values
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)

    #Training Models
    logger.info("Model Training Initiation\n=====================")
    kmeans = KMeans(random_state=0)
    mars_ = Earth(feature_importance_type='gcv',)
    vigilance_x = np.linspace(0.01, 3, Config.vigilance_x_frequency)
    for sens_x in vigilance_x:
        logger.info("Sensitivity Level {}".format(sens_x))
        lsnr = PR(mars_,vigil_x=sens_x)
        lsnr.fit(X_train,y_train)
        for i in range(5):
            q = train_df.iloc[i].values[:4].reshape(1,-1)
            q = sc.transform(q)
            start = time.time()
            m = lsnr.get_model(q)
            end = time.time()
            data['vigil_x'] = sens_x
            data['explanation_serving_time'] = end
            data['l1'] = lsnr.get_number_of_l1()
            data['l2'] = lsnr.get_number_of_l2()
    return data

def explanation_serving_t(train_df):
    data = {'vigil_t' : [], 'explanation_serving_time':[], 'l1':[],'l2':[]}
    X_train = train_df[['x','y','x_range','y_range']].values
    y_train = train_df['count'].values
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)

    #Training Models
    logger.info("Model Training Initiation\n=====================")
    kmeans = KMeans(random_state=0)
    mars_ = Earth(feature_importance_type='gcv',)
    vigilance_t = np.linspace(0.01, 3, Config.vigilance_t_frequency)
    for sens_t in vigilance_t:
        logger.info("Sensitivity Level {}".format(sens_x))
        lsnr = PR(mars_,vigil_theta=sens_t)
        lsnr.fit(X_train,y_train)
        for i in range(5):
            q = train_df.iloc[i].values[:4].reshape(1,-1)
            q = sc.transform(q)
            start = time.time()
            m = lsnr.get_model(q)
            end = time.time()
            data['vigil_t'] = sens_x
            data['explanation_serving_time'] = end
            data['l1'] = lsnr.get_number_of_l1()
            data['l2'] = lsnr.get_number_of_l2()
    return data

def prediction_serving_time(train_df):
    data = {'dimensionality' : [], 'prediction_serving_time':[], 'l1':[],'l2':[]}
    X_train = train_df[['x','y','x_range','y_range']].values
    y_train = train_df['count'].values
    for i in range(5):
        X_train = np.column_stack((X_train, X_train))
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        kmeans = KMeans(random_state=0)
        mars_ = Earth(feature_importance_type='gcv',)
        lsnr = PR(mars_,vigil_theta=sens_t)
        lsnr.fit(X_train,y_train)
        for j in range(5):
            q = X_train[j,:].reshape(1,-1)
            start = time.time()
            m = lsnr.get_model(q).predict(q)
            end = time.time()
            data['dimensionality'] = q.shape[1]
            data['prediction_serving_time'] = end
            data['l1'] = lsnr.get_number_of_l1()
            data['l2'] = lsnr.get_number_of_l2()

    return data
if __name__=='__main__':
    np.random.seed(15)
    logger.info("Finding datasets...")
    train_df = pd.read_csv('/home/fotis/dev_projects/explanation_framework/input/Crimes_Workload/train_workload_x-gauss-length-gauss-5-users-50000.csv', index_col=0)
    logger.info("Beginning Training Time Performance Measurement")
    data = training_time(train_df)
    eval_df = pd.DataFrame(data)
    eval_df.to_csv('output/Performance/evaluation_results_training_time.csv')
    logger.info("Beginning Explanation Serving Performance Measurement on vigil x")
    data = explanation_serving_x(train_df)
    eval_df = pd.DataFrame(data)
    eval_df.to_csv('output/Performance/explanation_serving_x.csv')
    logger.info("Beginning Explanation Serving Performance Measurement on vigil t")

    data = explanation_serving_t(train_df)
    eval_df = pd.DataFrame(data)
    eval_df.to_csv('output/Performance/explanation_serving_t.csv')
    logger.info("Beginning Prediction Serving Performance Measurement")
    data = prediction_serving_time(train_df)
    eval_df = pd.DataFrame(data)
    eval_df.to_csv('output/Performance/prediction_serving_time.csv')
