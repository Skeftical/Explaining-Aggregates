from sklearn.cluster import KMeans
import numpy as np
import copy
from pyearth import Earth

class OnlineMode():
    def partial_fit(self, q):
        dist = np.linalg.norm(self.cluster_centers - q[:-2],axis=1) #Calculate distances between all neurons
        closest_x = np.argmin(dist)
        #Scenario 2 Go to closest X and then to theta with lowest error
        dist_t = np.linalg.norm(self.theta_centers_copy[closest_x] - q[-2],axis=1)
        preds = []
        for m in self.new_models[closest_x]:
            preds.append(float(self.new_models[closest_x][m][-1].predict([q[-2]])))

        preds = np.array(preds)
        errors = np.sqrt((preds-q[self.TARGET_VARIABLE])**2) #Error between prediction and each model (y-y_hat)**2
        #Normalize errors
        errors = (errors-self.min_tot_data) / (self.max_tot_data-self.min_tot_data)

        total_distance = self.LAMBDA*dist_t +  (1-self.LAMBDA)*errors # Distance from x  + distance from theta cluster
        closest_t = np.argmin(total_distance)
        closest = (closest_x, closest_t)
        #Adding to cluster
        self.subclusters_copy[closest[0]][closest[1]] = np.row_stack((self.subclusters_copy[closest[0]][closest[1]],q))
        #Readjust theta
        try:
            self.theta_centers_copy[closest[0]][closest[1]] += self.ALPHA*(q[2] - self.theta_centers_copy[closest[0]][closest[1]])
        except IndexError, e:
            print(total_distance)
            print(closest)
            print(dist_t.shape)
            print(len(self.theta_centers_copy[closest[0]][0]))
            print(e)
            raise IndexError
        #Retraining every n_s steps
        self.affected.append(closest)
        if (self.t%self.n_s)==0:
            for tpl in set(self.affected):
                model = Earth( max_terms=20, enable_pruning=False, allow_linear=False)
                xtrain_o = self.subclusters_copy[tpl[0]][tpl[1]][:,-2]
                ytrain_o = self.subclusters_copy[tpl[0]][tpl[1]][:,self.TARGET_VARIABLE]
                try :
                    self.new_models[tpl[0]][tpl[1]].append(model.fit(xtrain_o, ytrain_o)) #Append a new model
                except ValueError, e:
                    print(tpl)
                    print(e)
                    raise ValueError
            affected = []
        self.t+=1

    def get_cluster_centers(self):
        return self.cluster_centers

    def get_theta_centers(self):
        return self.theta_centers_copy

    def get_models(self):
        return self.new_models

    def get_subclusters(self):
        return self.subclusters_copy

    def __init__(self,sample_data, target, alpha, l, new_clusters, theta_centers, cluster_centers, final_product,no_qs=20):
        self.TARGET_VARIABLE = target
        self.min_tot_data = np.min(sample_data[:,self.TARGET_VARIABLE].astype(np.float64))
        self.max_tot_data = np.max(sample_data[:,self.TARGET_VARIABLE].astype(np.float64))
        self.ALPHA = alpha
        self.LAMBDA = l
        self.subclusters_copy = copy.deepcopy(new_clusters)
        self.theta_centers_copy = copy.deepcopy(theta_centers) #theta positions (multiple for each clusterhead)
        self.cluster_centers = copy.copy(cluster_centers) # x posistions of clusterheads
        self.new_models = copy.deepcopy(final_product)
        self.n_s = no_qs
        self.t = 1 # every 20 , retrain ONLY affected MARS models
        self.affected = [] #Keep a list of affected models to retrain only them
