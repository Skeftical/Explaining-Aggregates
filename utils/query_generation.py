from pathlib import Path
import numpy as np
import os
import sys
import itertools
from pathlib import Path
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,)
logger = logging.getLogger("__main__")
def generate_boolean_vector(f,q,r,DIMS):
    """
    Generate boolean vector to filter dataset

    Parameters:
    -----------
    f : ndarray
    data to which the query is executed
    q : ndarray
    multi-dimensional point represent x
    r : ndarray
    multi-dimensional vector representing l
    DIMS: int
    number of dimensions
    """
    b = None
    for i in range(DIMS):
        if b is None:
            b =  (f[:,i]<q[i]+r[i]) & (f[:,i]>q[i])
        else :
            b = b & (f[:,i]<q[i]+r[i]) & (f[:,i]>q[i])
    return b

#different execution depending on aggr or density
#Execute Query Function
def execute_query_dens(b,data_space):
    res = data_space[b]
    return res.shape[0]
def execute_query_aggr(b,data_space):
    res = data_space[b]
    return float(np.mean(res[:,-1])) if res.shape[0]!=0 else 0

def generate_queries(DIMS,t,data):
    """
    Generates queries of arbitrary dimension over given data

    Parmeters:
    ----------
    DIMS : int
    dimensionality of given dataset
    t : str
    'density' or 'aggr' to differentiate between COUNT aggregate and the rest of the aggregates
    data : ndarray
    The multi-dimensional dataset of row vectors
    """
    #Start With clusters
    x = np.linspace(0,1,6)
    a = [x.tolist()]*DIMS
    #Define cluster centers and covariance matrix
    cluster_centers  = list(itertools.product(*a))
    cov = np.identity(DIMS)*0.2
    logger.debug("Generating queries at %d cluster centers" % len(cluster_centers))
    query_centers = []
    #Generate queries over cluster centers
    for c in cluster_centers:
        queries = np.random.multivariate_normal(np.array(c), cov, size=50)
        query_centers.append(queries)
    query_centers = np.array(query_centers).reshape(-1,DIMS)

    ranges = np.random.uniform(low=0.03**(1/DIMS), high=0.15**(1/DIMS), size=(query_centers.shape[0], 1))
    ranges = np.ones((query_centers.shape[0], DIMS))*ranges
    assert(ranges.shape[0]==query_centers.shape[0])
    queries = []
    i=0
    logger.debug("Query Generation")
    for q,r in zip(query_centers,ranges):
        b = generate_boolean_vector(data,q,r,DIMS)
        if t=='density':
            qt = q.tolist()
            qt += r.tolist()
            qt.append(execute_query_dens(b,data))
            queries.append(qt)
        elif t=='aggr':
            qt = q.tolist()
            qt += r.tolist()
            qt.append(execute_query_aggr(b,data))
            queries.append(qt)
        i+=1
    logger.debug("Generated {0} queries".format(len(queries)))
    return queries

if __name__=='__main__':
    #Generate Queries
    directory = os.fsencode('input')
    for file in os.listdir(directory):
        qs = []
        filename = os.fsdecode(file)
        if not filename.endswith(".csv") and filename.startswith("data"):
            a =filename.split('_')
            t = a[1]
            dim = int(a[2].split('=')[1])
            multi = a[-1]
            #Check if query file has been generated and skip
            qf = 'input/queries/queries-uniform-{0}-multi_{1}-{2}'.format(dim,multi,t)
            if Path(qf).exists():
                print("Query file '{0}' already exists skipping ".format(qf))
                continue;
            logger.debug('Loading file')
            f = np.loadtxt('input/%s' % (filename) ,delimiter=',')
            logger.debug("Loaded file with filename %s" % (filename))
            if t=='aggr':
                f = f.reshape(-1, dim+1)
            else:
                f = f.reshape(-1, dim)
            qs = generate_queries(dim,t,f)
            logger.debug("Current shape {0}".format(len(qs)))
            qs = np.array(qs).reshape(-1, 2*dim+1)
            logger.debug("New shape {0}".format(qs.shape))
            np.savetxt('input/queries/queries-uniform-{0}-multi_{1}-{2}'.format(dim,multi,t),qs, delimiter=',')
    
