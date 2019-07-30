import pandas as pd
import numpy as np
import os
import sys

os.chdir("../../../explanation_framework")
sys.path.append("../explanation_framework")
sys.path.append('utils')

from terminal_outputs import printProgressBar
from confs import Config
from setup import logger
from joblib import Parallel, delayed

if not os.path.exists('input/Subqueries'):
        logger.info('creating directory Subqueries')
        os.makedirs('input/Subqueries')
existing = set(os.listdir('input/Subqueries'))
DIM = 2
NSUBQUERIES = Config.NSUBQUERIES

def load_data():
    logger.info("Loading Data...")
    data = pd.read_csv('input/Crimes_-_2001_to_present.csv', header=0)
    dd = data[['X Coordinate', 'Y Coordinate', 'Arrest', 'Beat']]
    global dd_matrix
    dd_matrix = dd.values
    dd_matrix[:,2] = dd_matrix[:,2].astype(int)

def vectorize_query(q):
    res = dd_matrix[np.all((dd_matrix[:,:2]>q[:,:DIM]-q[:,DIM:2*DIM]) & (dd_matrix[:,:2]<q[:,:DIM]+q[:,DIM:2*DIM]),axis=1)]
    return np.array([res.shape[0], np.sum(res[:,2]),np.mean(res[:,3])]) if res.shape[0]!=0 else np.zeros(3)

def get_pertubations(sq):
    pertubation = np.random.uniform(low=-1,high=1, size=(NSUBQUERIES,2*DIM))*(0.01*std)
    S = sq.reshape(1,7)[:,:4]+pertubation
    res_matrix = np.array(list(map(lambda q : vectorize_query(q.reshape(1,4)), S)))
    assert res_matrix.shape == (NSUBQUERIES,3)
    subqueries = np.column_stack((pertubation,res_matrix))
    return subqueries


def generate_subqueries_for_files():
    directory = os.fsencode('input/Crimes_Workload')
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('test') and 'subqueries_{}'.format(filename) not in existing:
            logger.info("Loading Workload : {}".format(filename))
            df = pd.read_csv('input/Crimes_Workload/{}'.format(filename), index_col=0)
            global std
            std = df[['x','y','x_range','y_range']].std().values
            pertubations = Parallel(n_jobs=4, verbose=2)(delayed(get_pertubations)(sq)
                                                                                for sq in df.values[:1000,:])
            pertubations = np.array(pertubations)
            logger.info("Saving file {}".format(filename))
            np.save('input/Subqueries/subqueries_{}'.format(filename),pertubations)
        else:
            logger.info("Skipping {}".format(filename))

if __name__=='__main__':
    np.random.seed(15)
    load_data()
    generate_subqueries_for_files()
