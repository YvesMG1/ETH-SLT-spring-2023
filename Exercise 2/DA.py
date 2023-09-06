import sklearn as skl
from sklearn.utils.validation import check_is_fitted
from scipy.spatial import distance_matrix

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y



class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, T = 200, T_min = 3, alpha = 0.95, beta = 1e-4, noise = 0.05, metric="euclidian"):
        self.n_clusters = n_clusters
        self.K = 1
        self.random_state = random_state
        self.metric = metric
        self.T = T
        self.T_min = T_min
        self.noise = noise
        self.alpha = alpha
        self.beta = beta

        self.cluster_centers = None
        self.cluster_probs = None
        self.marg_probs = None

        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()
        self.bifurcation_tree = Tree()
        self.bifurcation_tree_cut_idx = None

    
    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        n_samp, n_clusters = dist_mat.shape[0], dist_mat.shape[1]
        probs = np.zeros((n_samp, n_clusters))

        norms = np.sum(self.marg_probs[:n_clusters] * np.exp(-np.square(dist_mat) / temperature), axis=1)
        probs = self.marg_probs[:n_clusters] * np.exp(-np.square(dist_mat) / temperature) / norms[:, np.newaxis]
        
        return probs

    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """

        D = distance_matrix(samples, clusters)

        return D

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (samples, n_features)
        """


        # Get number of features and number of samples in dataset
        n_samp = samples.shape[0]
        n_feat = samples.shape[1]

        # Initialize first cluster center as mean
        initial_cluster_center = np.zeros(shape=(self.K, n_feat))
        initial_cluster_center[0] = samples.mean(axis = 0)
        self.cluster_centers = initial_cluster_center
        
        # Root node of tree
        self.bifurcation_tree.create_node(identifier=0, data={'cluster_id': 0, 'distance': list(), 
                                                              'direction': 'right'})
        idx = 0 # cut idx
        
        # Set initial marginal probs and
        self.marg_probs = np.ones(shape=self.n_clusters)
        
        
        while self.T > self.T_min:
            

            # Update assignment probs and cluster centers until convergence
            while True:
                
                old_centers = self.cluster_centers.copy()
                
                # Calculate distance matrix and assignment probs
                D = self.get_distance(samples, self.cluster_centers)
                self.cluster_probs = self._calculate_cluster_probs(D, self.T)
                
                # Update cluster centers based on new assignment probs
                for j in range(self.K):
                    self.marg_probs[j] = np.sum(self.cluster_probs[:, j], axis = 0) / n_samp
                    self.cluster_centers[j] = np.sum(samples * self.cluster_probs[:, j, None], axis = 0) / self.marg_probs[j] / n_samp
                
                # Check if distance between previous centers is below threshhold
                if np.linalg.norm(old_centers - self.cluster_centers) < self.beta:
                    break
            
            
            # Update distances in bifurcation tree
            for j in range(self.K):
                
                # Calculate new distance to center
                new_dist = np.linalg.norm(initial_cluster_center - self.cluster_centers[j])
                
                cluster_node = self.bifurcation_tree.get_node(j)
                
                # Update distance 
                # Differentiate between left and right direction
                if cluster_node.data['direction'] == 'left':
                    cluster_node.data['distance'].append(-new_dist)
                    self.bifurcation_tree.update_node(nid=j, 
                                                      data = {'cluster_id': j, 
                                                              'distance': cluster_node.data['distance'],
                                                              'direction': cluster_node.data['direction']})

                else:
                    cluster_node.data['distance'].append(new_dist)
                    self.bifurcation_tree.update_node(nid=j, 
                                                      data = {'cluster_id': j, 
                                                              'distance': cluster_node.data['distance'],
                                                              'direction': cluster_node.data['direction']})
            
            k = self.K
            # Check for critical temperature
            if self.K < self.n_clusters:
                for j in range(k):
                    
                    # Calculate cova matrix for critical temp
                    cov_mat = np.zeros((n_feat, n_feat))
                    for i in range(n_samp):
                        diff = samples[i] - self.cluster_centers[j]
                        outer_product = np.outer(diff, diff)
                        weighted_outer = outer_product * self.cluster_probs[i, j] / self.marg_probs[j]
                        cov_mat += weighted_outer
                    cov_mat /= n_samp
                   
                    # Critical temp
                    crit_T = 2*abs(max(np.linalg.eig(cov_mat)[0]))


                    # Check if current temp is below critical temp
                    if self.T < crit_T:

                        # Create new cluster center
                        parent_center = np.copy(self.cluster_centers[j])
                        new_center = parent_center
                        self.cluster_centers = np.vstack((self.cluster_centers, new_center))
                        self.K += 1

                        # Create parent note from cluster moving to right side
                        parent_node = self.bifurcation_tree.get_node(j) # get data from parent
                        self.bifurcation_tree.update_node(j, identifier = 'parent') # transform initial node as parent node

                        # Create first child as old cluster
                        self.bifurcation_tree.create_node(identifier=j, parent = 'parent',
                                                          data = {'cluster_id': j, 
                                                                  'distance': parent_node.data['distance'],
                                                                  'direction': 'right'})
                        
                        # Create second child as new cluster moving to left side
                        dist_list = []
                        dist_list.append(parent_node.data['distance'][-1].item())
                        self.bifurcation_tree.create_node(identifier=self.K-1, parent = 'parent',
                                                          data = {'cluster_id': self.K-1,
                                                                  'distance': dist_list,
                                                                  'direction': 'left'})
                        
                        # Set unique id for parent 
                        self.bifurcation_tree.update_node('parent', identifier = 'parent: ' + str(j) +str(self.K-1))
                        
                        # Update cut idx
                        self.bifurcation_tree_cut_idx = idx
                        
                        if self.K == self.n_clusters:
                            break

        
            
            # calculate distortion
            distortion = np.sum(np.square(D) * self.cluster_probs) / n_samp
            
            # Add current distortion, temperature and # of clusters to list
            self.distortions.append(distortion)
            self.temperatures.append(self.T)
            self.n_eff_clusters.append(self.K)

            # Add small amount of noise to cluster centers
            self.cluster_centers += np.random.normal(0, self.noise, (self.K, n_feat))
            
            # Reduce temperature
            self.T = self.T * self.alpha
            
            # Update cut_idx
            idx +=1
            
        self.bifurcation_tree.show()

        return self
            


    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """
        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(self.K)]
        beta = [1 / t for t in self.temperatures]
        
        
        for leave in self.bifurcation_tree.leaves():
            diff = len(beta) - len(leave.data['distance'])
            clusters[leave.data['cluster_id']] = [np.nan for _ in range(diff)] + leave.data['distance']
  
            
        # Cut the last iterations, usually it takes too long
        cut_idx = self.bifurcation_tree_cut_idx + 20

        
        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):

            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()
