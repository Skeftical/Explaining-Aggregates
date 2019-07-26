import pandas as pd
import numpy as np
import os
import sys

os.chdir("../../../explanation_framework")
sys.path.append('utils')
from terminal_outputs import printProgressBar
from confs import Config
QUERIES = Config.chicago_queries
CLUSTERS = Config.chicago_clusters

if not os.path.exists('input/Crimes_Workload'):
        print('creating directory' ,'Crimes_Workload')
        os.makedirs('input/Crimes_Workload')
existing = set(os.listdir('input/Crimes_Workload'))
if 'train_workload_gauss-{}-users-{}.csv'.format(CLUSTERS,QUERIES) in existing or 'test_workload_gauss-{}-users-{}.csv'.format(CLUSTERS,QUERIES) in existing:
    print("Files already exist; exiting ")
    sys.exit(0)

def load_data():
    print("Loading Data...")
    global df
    df = pd.read_csv('input/Crimes_-_2001_to_present.csv', header=0)
    sample = df.sample(10000)
    global x_mean
    x_mean = float(sample[['X Coordinate']].mean())
    global y_mean
    y_mean = float(sample[['Y Coordinate']].mean())
    global x_std
    x_std = float(sample[['X Coordinate']].std())
    global y_std
    y_std = float(sample[['Y Coordinate']].std())
    global x_min
    x_min = float(sample[['X Coordinate']].min())
    global y_min
    y_min = float(sample[['Y Coordinate']].min())
    global x_max
    x_max = float(sample[['X Coordinate']].max())
    global y_max
    y_max = float(sample[['Y Coordinate']].max())
    del sample

def set_number_and_locations_of_clusters_Normal():
    print("Constructing central points for queries")
    clusters = np.random.multivariate_normal([x_mean, y_mean], [[x_std**2, 0],[0, y_std**2]], CLUSTERS)
    #Create queries around those central points
    #Fraction of variance for how widespread the queries would be
    queries = map(lambda x : np.random.multivariate_normal(x, [[0.01*(x_std),0],[0,0.01*(y_std)]], int(QUERIES/CLUSTERS)), clusters)
    col_queries = np.array(list(queries)).reshape(-1,2)
    return col_queries

def set_number_and_locations_of_clusters_Uniform():
    print("Constructing central points for queries")
    clusters = np.random.uniform(low=(x_min,y_min), high=(x_max,y_max),size=(CLUSTERS,2))
    #Create queries around those central points
    #Fraction of variance for how widespread the queries would be
    length = np.array([0.01*(x_std), [0.01*(y_std)]])
    queries = map(lambda x : np.random.uniform(low=x-length,high=x+length , size=int(QUERIES/CLUSTERS)), clusters)
    col_queries = np.array(list(queries)).reshape(-1,2)
    return col_queries

def __get_query(q):
    multiplier_x = np.random.rand()#varying range at queries
    x_range = (x_std/2.0)*multiplier_x
    predicate1_0 = df['X Coordinate']>=q[0]-x_range
    predicate1_1 = df['X Coordinate']<=q[0]+x_range
    multiplier_y = np.random.rand()
    y_range = (y_std/2.0)*multiplier_y
    predicate2_0 = df['Y Coordinate']>=q[1]-y_range
    predicate2_1 = df['Y Coordinate']<=q[1]+y_range
    return (x_range, y_range, ((predicate1_0) & (predicate1_1) & (predicate2_0) & (predicate2_1)))

def __get_query_normal(q):
    x_range = np.random.normal((x_std/2.0), (x_std/2.0)*np.random.rand())#varying range at queries
    predicate1_0 = df['X Coordinate']>=q[0]-x_range
    predicate1_1 = df['X Coordinate']<=q[0]+x_range
    y_range = np.random.normal((y_std/2.0), (y_std/2.0)*np.random.rand())#varying range at queries
    predicate2_0 = df['Y Coordinate']>=q[1]-y_range
    predicate2_1 = df['Y Coordinate']<=q[1]+y_range
    return (x_range, y_range, ((predicate1_0) & (predicate1_1) & (predicate2_0) & (predicate2_1)))

def construct_queries(col_queries,xdist, ldist):
    complete_queries = []
    i=0
    for q in col_queries:
        if ldist=="gauss":
            call = __get_query_normal(q)
        else:
            call = __get_query(q)
        x_range = call[0]
        y_range = call[1]
        res = df[call[2]]
        count = int(res.count()[0])
        sum_ = float(res['Arrest'].sum())
        avg = float(res['Beat'].mean())
        complete_queries.append([q[0], q[1],x_range, y_range, count, sum_, avg])
        i+=1
        printProgressBar(i, QUERIES,prefix = 'Progress:', suffix = 'Complete', length = 50)
    finished = pd.DataFrame(np.array(complete_queries), columns=['x','y','x_range', 'y_range', 'count', 'sum_','avg'])
    test = finished.sample(frac=.2)
    train = finished.drop(test.index)
    print("Saving Output Files")
    test['count'] = test['count'].replace([np.inf, -np.inf], np.nan).dropna()
    test['sum_'] = test['sum_'].replace([np.inf, -np.inf], np.nan).dropna()
    train['count'] = train['count'].replace([np.inf, -np.inf], np.nan).dropna()
    train['sum_'] = train['sum_'].replace([np.inf, -np.inf], np.nan).dropna()
    test.dropna().to_csv('input/Crimes_Workload/test_workload_x-{}-length-{}-{}-users-{}.csv'.format(xdist, ldist, CLUSTERS,QUERIES))
    train.dropna().to_csv('input/Crimes_Workload/train_workload_x-{}-length-{}-{}-users-{}.csv'.format(xdist, ldist, CLUSTERS,QUERIES))

if __name__=='__main__':
    np.random.seed(15)
    load_data()
    distributions = ["gauss", "uniform"]
    for xdist in distributions:
        for ldist in distributions:
            if xdist=="gauss":
                col_queries = set_number_and_locations_of_clusters_Normal()
            else:
                col_queries = set_number_and_locations_of_clusters_Uniform()
            construct_queries(col_queries, xdist, ldist)
