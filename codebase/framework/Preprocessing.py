from sklearn.cluster import KMeans
import numpy as np
import copy
from pyearth import Earth
from setup import logger

class PreProcessing():

    def __preprocessing_x(self,X,y, vigil=.05):
        # #Tuning k-parameter for kmeans
        c=0
        prev_inertia = 0
        pres_inertia = 0
        init = True
        diff = np.inf
        X_ = X[:,:self.d//2]
        logger.info("Shape of X in preprocessing x is : {}".format(X_.shape))
        while diff>= vigil:
            c+=1
            kmeans = KMeans(n_clusters=c)
            kmeans.fit(X_)
            pres_inertia = kmeans.inertia_
            if not init:
                diff = np.log(np.abs(prev_inertia - pres_inertia))
                prev_inertia = pres_inertia
            else:
                prev_inertia = pres_inertia
                init=False
        # #End of tuning
        logger.info("Number of clusters in X : {}".format(c))
        CLUSTERS = c
        kmeans = KMeans(n_clusters=CLUSTERS)
        kmeans.fit(X_)

        #Assigning to clusters
        for i in np.unique(kmeans.labels_):
            mask = np.where(kmeans.labels_== i)
            self.data_in_clusters_L1[i] = np.column_stack((X[mask], y[mask]))
            logger.info("Data shape in cluster {} : {}".format(i,self.data_in_clusters_L1[i].shape))
        self.CLUSTER_CENTERS = kmeans.cluster_centers_
        logger.info("Cluster centers shape {}".format(self.CLUSTER_CENTERS.shape))


    def __preprocessing_theta(self, vigil=.4):
        import warnings
        warnings.filterwarnings('ignore')

        #For each cluster
        for j in self.data_in_clusters_L1:
            #-2 Since each vector is made up of x\inR^d and y
            X = self.data_in_clusters_L1[j][:,self.d//2:-1]#Only care about clustering thetas
            logger.info("Shape of theta vector {}".format(X.shape))
            c = 1
            if (X.shape[0]>10):
        #         raise ValueError("Error in support of cluster")
            #Tuning k-parameter for kmeans
                c=0
                prev_inertia = 0
                pres_inertia = 0
                init = True
                diff = np.inf
                while np.log(diff)>= vigil:
                    c+=1
                    t_kmeans = KMeans(n_clusters=c)
                    t_kmeans.fit(X)
                    pres_inertia = t_kmeans.inertia_
                    if not init:
                        diff = np.log(np.abs(prev_inertia - pres_inertia))
                        prev_inertia = pres_inertia
                    else:
                        prev_inertia = pres_inertia
                        init=False

                if np.unique(t_kmeans.labels_).shape[0]!=len(t_kmeans.cluster_centers_):
                    print("Cluster {0}".format(j))
                    c = c-1
            logger.info("Number of clusters in thetas {}".format(c))
        #     #End of tuning

            CLUSTERS = c
            t_kmeans = KMeans(n_clusters=CLUSTERS)
            t_kmeans.fit(X)

            for i in range(CLUSTERS):
                    mask = np.where(t_kmeans.labels_== i)[0]
                    logger.info("Data shape in cluster L1 {}, L2 {} : {}".format(j,i,self.data_in_clusters_L1[j][mask].shape))
                    self.data_in_clusters_L2[(j,i)] = self.data_in_clusters_L1[j][mask]
            self.THETA_CENTERS[j]= t_kmeans.cluster_centers_
            logger.info("Shape of theta clusters in L1 {} : {}".format(j, self.THETA_CENTERS[j].shape))

    def __fit_models(self):
        #Fit an Earth model for each cluster
        for l1,l2 in self.data_in_clusters_L2:
            tcluster = self.data_in_clusters_L2[(l1,l2)]
            XX = tcluster[:,:self.d]
            logger.info("Shape of Training data {}".format(XX.shape))
            yy = tcluster[:,-1]
            try:
                estimator = deepcopy(self.learning_algorithm)

                # model = Earth(max_degree=1, feature_importance_type='gcv')
                estimator.fit(XX,yy)
            except ValueError as e:
                print((i,j))
                print(e)
                raise ValueError

            self.final_product[(l1,l2)] = model

    def fit(self,X,y):
        self.d = X.shape[1]
        self.min_y = np.min(y)
        self.max_y = np.max(y)
        self.__preprocessing_x(X,y, self.vigil_x)
        self.__preprocessing_theta(self.vigil_theta)
        self.__fit_models()

    def get_cluster_centers(self):
        return self.CLUSTER_CENTERS

    def get_theta_centers(self):
        return self.THETA_CENTERS

    def get_models(self):
        return self.final_product

    def get_closest_l1(self, q):
        assert q[:,:self.d//2].shape[1]==self.CLUSTER_CENTERS.shape[1]
        dist = np.linalg.norm(self.CLUSTER_CENTERS - q[:,:self.d//2],axis=1) #Calculate distances between all neurons
        closest_x = np.argmin(dist)
        return closest_x

    def get_closest_l2(self, q):
        cx = self.get_closest_l1(q) #Obtain L1 representative
        tc = self.get_theta_centers()[cx] #Obtain L2 associated theta Centers
        dist = np.linalg.norm(tc - q[:,self.d//2:],axis=1) #Calculate distances between all neurons
        closest_t = np.argmin(dist)
        return (cx, closest_t)

    def get_model(self,q):
        return self.final_product[self.get_closest_l2(q)]


    def __init__(self, learning_algorithm, vigil_theta=.4, vigil_x=.05):
        '''
        arg:
            d : dimensionality
        '''
        self.d = None
        self.vigil_theta=vigil_theta
        self.learning_algorithm = learning_algorithm
        self.vigil_x = vigil_x
        self.data_in_clusters_L1 = {}
        self.CLUSTER_CENTERS = None
        self.data_in_clusters_L2 = {}
        self.THETA_CENTERS = {}
        self.final_product = {}


class OnlineMode(PreProcessing):


        def partial_fit(self, X, y):
            assert X.shape[1]==self.d
            cx = self.get_closest_l1(q) #Obtain L1 representative
            tc = self.get_theta_centers()[cx] #Obtain L2 associated theta Centers
            dist = np.linalg.norm(tc - q[:,self.d//2:],axis=1) #Calculate distances between all neurons
            closest_t = np.argmin(dist)
            #Get All get_models for closest L1
            assoc_models = filter(lambda x: x[0]==cx, self.final_product)
            preds = np.array([self.final_product[k].predict(X) for k in assoc_models])
            errors = np.sqrt((preds-y)**2) #Error between prediction and each model (y-y_hat)**2
            #Normalize
            errors = (errors.self.min_y)/(self.max_y-self.min_y)
            combined_distances = 0.5*dist + 0.5*errors
            ct = np.argmin(combined_distances)
            assert combined_distances.shape[0] == self.THETA_CENTERS[cx].shape[0]
