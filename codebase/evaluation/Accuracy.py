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
from sklearn.neighbors import KernelDensity
from sklearn import metrics
from scipy.stats import entropy
from codebase.framework.Preprocessing import PreProcessing as PR
import argparse
import logging


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
if not os.path.exists('output/Accuracy'):
        logger.info('creating directory Accuracy')
        os.makedirs('output/Accuracy')




def kl_divergence_error(y, y_hat):
    kd = KernelDensity(bandwidth=0.75).fit(y.reshape(-1,1))
    yp = kd.score_samples(y.reshape(-1,1))
    kd = KernelDensity(bandwidth=0.75).fit(y_hat.reshape(-1,1))
    ypg = kd.score_samples(y_hat.reshape(-1,1))
    return entropy(yp,ypg)

def model_based_divergence(X,y, model_2):
    model_1 = Earth(feature_importance_type='gcv')
    model_1.fit(X,y)
    features_l = model_1.feature_importances_.reshape(-1,1)
    features_else = model_2.feature_importances_.reshape(-1,1)
    return metrics.pairwise.cosine_similarity(features_l, features_else)

def metrics_for_model(model_name, dataset_name,aggregate_name,y_hat,X, y,model, res_eval):
    r2 = metrics.r2_score(y,y_hat)
    kl = kl_divergence_error(y, y_hat)
    md = model_based_divergence(X,y,model) #model comparison with itself
    nrmse = np.sqrt(metrics.mean_squared_error(y, y_hat))/np.mean(y)
    res_eval['model'].append(model_name)
    res_eval['aggregate_name'].append(aggregate_name)
    res_eval['dataset'].append(dataset_name)
    res_eval['kl'].append(kl)
    res_eval['r2'].append(r2)
    res_eval['md'].append(md)
    res_eval['nrmse'].append(nrmse)




if __name__=='__main__':
    np.random.seed(15)
    logger.info("Finding datasets...")
    directory = os.fsencode('input/Crimes_Workload')
    directory_sub = os.fsencode('input/Subqueries/')
    patterns = {'gauss-gauss': '*x-gauss*-length-gauss*',
               'gauss-uni': '*x-gauss*-length-uniform*',
               'uni-gauss': '*x-uniform*-length-gauss*',
               'uni-uni': '*x-uniform*-length-uniform*',}
    train_datasets = {}
    test_datasets = {}
    sub_datasets = {}

    for p in patterns:
        res = [os.fsdecode(n) for n in os.listdir(directory) if fnmatch.fnmatch(os.fsdecode(n), patterns[p])]
        train_datasets[p] = res[0] if res[0].startswith('train') else res[1]
        test_datasets[p] = res[0] if res[0].startswith('test') else res[1]
        sub_datasets[p] = [os.fsdecode(n) for n in os.listdir(directory_sub) if fnmatch.fnmatch(os.fsdecode(n), patterns[p])][0]

    res_eval = {'model': [],
               'dataset': [],
               'aggregate_name': [],
               'kl': [],
               'r2':[],
               'md':[],
               'nrmse':[]}
    #Main
    for p in patterns:
        logger.info('Beginning Evaluation for {0}'.format(p))
        logger.info('Loading Datasets...')

        test_df = pd.read_csv('/home/fotis/dev_projects/explanation_framework/input/Crimes_Workload/{0}'.format(test_datasets[p]), index_col=0)
        train_df = train = pd.read_csv('/home/fotis/dev_projects/explanation_framework/input/Crimes_Workload/{0}'.format(train_datasets[p]), index_col=0)
        sub = np.load('/home/fotis/dev_projects/explanation_framework/input/Subqueries/{0}'.format(sub_datasets[p]))

        logger.info('Finished loading\nCommencing Evaluation')
        aggregates = ['count','sum_','avg']
        for agg in aggregates:
            logger.info("Evaluating Aggregates : {0}".format(agg))
            X_train = train_df[['x','y','x_range','y_range']].values
            y_train = train_df[agg].values
            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            #Training Models
            logger.info("Model Training Initiation\n=====================")
            kmeans = KMeans()
            mars_ = Earth(feature_importance_type='gcv',)

            lsnr = PR(mars_)
            lsnr.fit(X_train,y_train)

            mars_global = Earth(feature_importance_type='gcv')
            mars_global.fit(X_train, y_train)

            logger.info("Accuracy Evaluation on Test set\n=====================")
            for i in range(1000):
                #Obtain query from test-set
                dataset = p
                printProgressBar(i, 1000,prefix = 'Progress:', suffix = 'Complete', length = 50)

                q = test_df.iloc[i].values[:4].reshape(1,-1)
                q = sc.transform(q)
                #Obtain subquery pertubations for query q from test set
                q1 = sub[i]
                X = q1[:,:4]
                y = q1[:,4]
                X = sc.transform(X)
                # Train local model (Should be the best out of the 3)
                mars = Earth(feature_importance_type='gcv')
                mars.fit(X,y)
                y_hat = mars.predict(X)
                metrics_for_model('local',dataset,agg,y_hat,X, y, mars,res_eval)

                #Obtain metrics for our
                y_hat_s = lsnr.get_model(q).predict(X)
                metrics_for_model('ours',dataset,agg,y_hat_s,X,y,lsnr.get_model(q) ,res_eval)


                #Obtain metrics for global
                y_hat_g = mars_global.predict(X)
                metrics_for_model('global',dataset,agg,y_hat_g,X,y,mars_global,res_eval)
            break;
        break;
    eval_df = pd.DataFrame(res_eval)
    eval_df.to_csv('output/Accuracy/evaluation_results.csv')
