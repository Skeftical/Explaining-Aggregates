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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import entropy
from codebase.framework.Preprocessing import PreProcessing as PR
import argparse
import pickle
import logging
from query_generation import generate_boolean_vector

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument('-v',help='verbosity',dest='verbosity',action="store_true")
parser.add_argument("--crimes",dest="crimes", action="store_true")
parser.add_argument("--higgs",dest="higgs", action="store_true")
parser.add_argument("--accelerometer",dest="accelerometer", action="store_true")

args = parser.parse_args()

if args.verbosity:
   print("verbosity turned on")
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(logging.DEBUG)
   logger.addHandler(handler)
if not os.path.exists('output/Accuracy'):
        logger.info('creating directory Accuracy')
        os.makedirs('output/Accuracy')
if not (args.crimes or args.higgs or args.accelerometer):
    logger.info("No data set specified")
    sys.exit()


def kl_divergence_error(y, y_hat):
    kd = KernelDensity(bandwidth=0.75).fit(y.reshape(-1,1))
    yp = kd.score_samples(y.reshape(-1,1))
    kd = KernelDensity(bandwidth=0.75).fit(y_hat.reshape(-1,1))
    ypg = kd.score_samples(y_hat.reshape(-1,1))
    return entropy(yp,ypg)

def model_based_divergence(X,y, model_2):
    model_1 = LinearRegression()# Earth(feature_importance_type='gcv')
    model_1.fit(X,y)
    features_l = model_1.coef_
    features_else = model_2.coef_
    a_ = np.linalg.norm(features_l)
    b_ = np.linalg.norm(features_else)
    return np.dot(features_l,features_else)/(a_*b_)

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

def accuracy_on_crimes():
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
        train_df = pd.read_csv('/home/fotis/dev_projects/explanation_framework/input/Crimes_Workload/{0}'.format(train_datasets[p]), index_col=0)
        sub = np.load('/home/fotis/dev_projects/explanation_framework/input/Subqueries/{0}'.format(sub_datasets[p]))

        logger.info('Finished loading\nCommencing Evaluation')
        aggregates = ['count','sum_','avg']
        agg_map = {'count' :4, 'sum_':5, 'avg':6}
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
            lr = Ridge()

            lsnr = PR(lr)
            lsnr.fit(X_train,y_train)

            lr_global = LinearRegression()
            lr_global.fit(X_train, y_train)

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
                y = q1[:,agg_map[agg]]
                X = sc.transform(X)
                # Train local model (Should be the best out of the 3)
                lr = LinearRegression()
                lr.fit(X,y)
                y_hat = lr.predict(X)
                metrics_for_model('local',dataset,agg,y_hat,X, y, lr,res_eval)

                #Obtain metrics for our
                y_hat_s = lsnr.get_model(q).predict(X)
                metrics_for_model('ours',dataset,agg,y_hat_s,X,y,lsnr.get_model(q) ,res_eval)


                #Obtain metrics for global
                y_hat_g = lr_global.predict(X)
                metrics_for_model('global',dataset,agg,y_hat_g,X,y,lr_global,res_eval)
            logger.info("Finished Queries")
    eval_df = pd.DataFrame(res_eval)
    eval_df.to_csv('output/Accuracy/evaluation_results_linear.csv')

def accuracy_on_higgs():
    logger.info("Starting Accuracy Tests on Higgs")
    logger.info("================================")
    df = pd.read_csv('input/sample_higgs_0.01.csv', index_col=0)
    X = df[['m_bb','m_wwbb']].dropna().values
    y = df['label']
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0)
    X = (X-min_) / (max_-min_)
    data = np.column_stack((X,y))
    x = np.linspace(0.1,0.9,7)
    xx,yy = np.meshgrid(x,x)
    DIMS = X.shape[1]
    cov = np.identity(DIMS)*0.001
    cluster_centers = np.column_stack((xx.ravel(),yy.ravel()))
    query_centers = []
    #Generate queries over cluster centers
    for c in cluster_centers:
        queries = np.random.multivariate_normal(np.array(c), cov, size=40)
        query_centers.append(queries)
    query_centers = np.array(query_centers).reshape(-1,DIMS)

    ranges = np.random.uniform(low=0.005**(1/3), high=0.25**(1/3), size=(query_centers.shape[0], DIMS))
    queries = []
    empty = 0
    for q,r in zip(query_centers,ranges):
            b = generate_boolean_vector(data,q,r,2)
            res = data[b]
            if res.shape[0]==0:
                empty+=1

            ans = float(np.mean(res[:,-1])) if res.shape[0]!=0 else 0
            qt = q.tolist()
            qt += r.tolist()
            qt.append(ans)
            queries.append(qt)
    qs = np.array(queries).reshape(-1, 2*DIMS+1)
    X_train, X_test, y_train, y_test = train_test_split(
         qs[:,:qs.shape[1]-1], qs[:,-1], test_size=0.4, random_state=0)
    lr = LinearRegression()
    lsnr = PR(lr)
    lsnr.fit(X_train, y_train)
    y_hat = np.array([float(lsnr.get_model(x.reshape(1,-1)).predict(x.reshape(1,-1))) for x in X_test])
    r2 = metrics.r2_score(y_test,y_hat)
    kl = kl_divergence_error(y_test, y_hat)
    nrmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat))/np.mean(y_test)
    logger.info("R2 Score : {}\nNRMSE : {}\nKL-Divergence : {}".format(r2, kl, nrmse))
    #Linear Regression comparsion
    lr.fit(X_train, y_train)
    y_hat_lr = lr.predict(X_test)
    r2_lr = metrics.r2_score(y_test, y_hat_lr)
    kl_lr = kl_divergence_error(y_test, y_hat_lr)
    nrmse_lr = np.sqrt(metrics.mean_squared_error(y_test, y_hat_lr))/np.mean(y_test)
    logger.info("R2 Score : {}\nNRMSE : {}\nKL-Divergence : {}".format(r2_lr, kl_lr, nrmse_lr))
    dic = {}
    dic['LPM' ]= [('r2',r2), ('kl',kl), ('nrmse',nrmse)]
    dic['LR'] = [('r2',r2_lr), ('kl',kl_lr), ('nrmse',nrmse_lr)]
    #Polynomial regression comparsion
    for count, degree in enumerate(np.arange(3,10,2)):
         model = make_pipeline(PolynomialFeatures(degree), Ridge())
         model.fit(X_train, y_train)
         y_hat = model.predict(X_test)
         r2_p = metrics.r2_score(y_test,y_hat)
         kl_p = kl_divergence_error(y_test, y_hat)
         nrmse_p = np.sqrt(metrics.mean_squared_error(y_test, y_hat))/np.mean(y_test)
         dic["LR ({})".format(degree)] = [('r2',r2_p), ('kl',kl_p), ('nrmse',nrmse_p)]
         print("R2 for degree {} : {}".format(degree, metrics.r2_score(y_test, y_hat)))
    logger.info("==============================================")
    with open('output/Accuracy/multiple_methods_higgs.pkl', 'wb') as handle:
        pickle.dump(dic, handle)

if __name__=='__main__':
    np.random.seed(15)
    if args.crimes:
        accuracy_on_crimes()
    if args.higgs:
        accuracy_on_higgs()
