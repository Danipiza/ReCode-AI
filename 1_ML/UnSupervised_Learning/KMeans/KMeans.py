import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import time 
import random
import os
import sys
import math

utils_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(utils_dir)

import ml_utils # type: ignore





"""
Clustering algorithm, KMeans.
The objective is to categorize a population of data 
in a number of clusters

Args:
    k (int)                : Number of clusters.
    population (float[][]) : Population with all the individuals.    
    func (function)        : Euclidean or Manhattan distance.    
    print (bool)           : Boolean used to print information.
"""
class KMeans:


    def __init__(self, k, population, func, print):
        self.k = k          

        self.population =population
        self.d          =len(population[0])     
        self.n          =len(population)    

        self.func=func

        self.print=print        
        
        if self.k>self.n:
            print('Erorr: Value of K is bigger than the size of the population')
        
    

    """
    Compare two centroids.

    Args:
        a (float[]) : Centroid.
        b (float[]) : Centroid.
    """
    def compare_centroids(self, a, b):        
        n=len(a)

        for i in range(n):
            for j in range(self.d):
                if a[i][j]!=b[i][j]: return False
        
        return True

    """
    Execute function.
    
    Return:
        assignment (int[])    : Last calculated assignment.
        centroids (float[][]) : Last calculated centroids.
    """
    def execute(self):
        
        # -- Init --------------------------------------------------------------        
        # random initialization of the centroids 
        # pick self.k random individuals.
               
        dic={}         
        centroids=[]   
        for i in range(self.k):

            while True:
                rand=random.randint(0, self.n-1)
                if rand not in dic:
                    centroids.append(self.population[rand])                
                    dic[rand]=1
                    break
        
        
        assignment=[-1 for i in range(self.n)]
        iterations=0


        # while the centroids doesnt change
        while True: 
            iterations+=1
        
                
            # -- 1st PHASE: Assignment -----------------------------------------            
            for i in range(self.n):
                tmp=-1
                cluster=-1
                dist=float('inf')
                
                # compare the actual individual with each centroid                
                for j in range(self.k): 
                    tmp=self.func(self.population[i], centroids[j])
                    
                    if dist>tmp:
                        dist=tmp
                        cluster=j
                
                # assign the closest cluster to the actual individual (i-th) 
                assignment[i]=cluster 

            
            
            
            # -- 2nd PHASE: Update centroids -----------------------------------            
                
            # number of individuals assigned for each cluster
            cluster_size=[0 for _ in range(self.k)] 

            
            new_centroids=[[0 for _ in range(self.d)] for _ in range(self.k)]
            for i in range(self.n):
                for j in range(self.d): 
                    new_centroids[assignment[i]][j]+=self.population[i][j]
                cluster_size[assignment[i]]+=1

            

            if self.print==True:
                print('Population: {}'.format(self.population))
                print('Centroids: {}'.format(centroids))
                print('Cluster sizes: {}'.format(cluster_size))

            
            # calculate new centroids. for each cluster divide by the it size
            for i in range(self.k): 
                for j in range(self.d):  
                    if cluster_size[i]!=0: new_centroids[i][j]/=cluster_size[i]   

            if self.print==True: print('New centroids: {}'.format(new_centroids))

            
            # -- 3rd PHASE: Compare centroids ----------------------------------            
            if(self.compare_centroids(centroids,new_centroids)): break
            centroids=new_centroids # dont finalizes. update centroids

        
        return assignment, centroids
    
   
"""
Prints the calculated assignment of the given population.

Args:
    population (float[][]) : Population with all the individuals.
    assignment (int[])     : Assignment of categories for each individual.
    k (int)                : Number of clusters.
"""
def plot2D(population,assignment,k):
    
    # -- Preparation of the plot data ------------------------------------------
    # colors
    colors=['blue', 'red', 'green', 'black', 'pink', 'yellow', 'magenta', 
            'brown', 'darkgreen', 'gray', 'fuchsia', 'violet', 'salmon', 
            'darkturquoise', 'forestgreen', 'firebrick', 'darkblue', 
            'lavender', 'palegoldenrod', 'navy']
    n=len(population)
    
        
    x=[[]for _ in range(k)]
    y=[[]for _ in range(k)]

    for i in range(n):
        x[assignment[i]].append(population[i][0])
        y[assignment[i]].append(population[i][1])

    # add the dots with its respective colors
    for i in range(k): plt.scatter(x[i], y[i], color=colors[i])  


    
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Assignments')
    
    plt.show()

    

"""
GUI function. 2 columns of plots.
First column:
    - Elbow diagram 
    - Davies Bouldin index.

Second column: 
    - Assignment of the individuals with 
    the calculated best number of clusters

Args:
    k (int): Maximumn number of clusters stored.
    best (int): Best number of clusters.
    fits (float[]): Fitness value for each stored number of clusters.
    coefs (float[]): DB coefficient for each stored number of clusters.
    population (float[][]) : Population with all the individuals.
    assignment (int[])  : Assignment of categories for each individual.
"""
def GUI(k, best, fits, coefs, population, assignment):    
    
    # -- Preparation of the plot data ------------------------------------------
    # colors
    colors=['blue', 'red', 'green', 'black', 'pink', 'yellow', 'magenta', 
            'brown', 'darkgreen', 'gray', 'fuchsia', 'violet', 'salmon', 
            'darkturquoise', 'forestgreen', 'firebrick', 'darkblue', 
            'lavender', 'palegoldenrod', 'navy']

    # first column 
    x1=[i for i in range(1, k+1)]      
    x2=[i for i in range(2, k+1)] 
    
    # second column 
    n=len(population)
    x3=[[] for _ in range(k)]    
    y3=[[] for _ in range(k)]
    for i in range(n):
        x3[assignment[i]].append(population[i][0])
        y3[assignment[i]].append(population[i][1])

    
    # -- fig and grid ----------------------------------------------------------    
    fig =plt.figure(figsize=(10, 6))
    gs  =GridSpec(2, 2, figure=fig)

    # Plot 1 (first column, fisrt plot)
    ax1=fig.add_subplot(gs[0, 0])
    ax1.plot(x1, fits, color='b', linestyle='-')
    # dot with the best number of cluster
    ax1.scatter(best+1, fits[best], color='red')  
    ax1.set_xlabel('Clusters')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Elbow diagram')
    ax1.grid(True)

    # Plot 2 (first column, second plot)
    ax2=fig.add_subplot(gs[1, 0])
    ax2.plot(x2, coefs, color='b', linestyle='-') 
    # dot with the best number of cluster
    ax2.scatter(best+1, coefs[best-1], color='red')  
    ax2.set_xlabel('Clusters')
    ax2.set_ylabel('Coefficient')
    ax2.set_title('Davies Bouldin')
    ax2.grid(True)
    ax2.set_xlim(1, k)  

    # Plot 3 (second column)
    ax3=fig.add_subplot(gs[:, 1])    
    # plot all the dots
    for i in range(k): ax3.scatter(x3[i], y3[i], color=colors[i])
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_title('population')
    
    
    plt.tight_layout() 
    plt.show() 

"""
Calculate the best number of clusters.

Arg:
    DBs (float[]) : Davies-Bouldin indices.

Return:
    ret (int) : index for the best number of clusters.
"""
def calculate_best_DB(DBs):
    n=len(DBs)
    
    ret=0    
    val=DBs[0]   

    for i in range(1,n):
        if val>DBs[i]: 
            val=DBs[i]
            ret=i

    return ret          

"""
Davies-Bouldin index.

DB=(1/k)*Sum(i=1&&i!=j -> k)[max((avg_distance(i,centroid[i])+
                                  avg_distance(i,centroid[i])/
                                  (centroids_distance(i,j)))]

Args:
    population (float[][]) : Population with all the individuals.
    assignment (int[])     : Assignment of categories for each individual.
    k (int)                : Number of clusters.
    centroids (float[][])  : Centroids of each cluster.
"""
def davies_bouldin(population, k, assignment, centroids):
    ret=0.0           # davies bouldin index
    n=len(population) # number of individuals
    
    tmp=0.0
    
    avg_distance=[0.0 for _ in range(k)]    
    # number of individuals for each cluster
    cluster_size=[0 for _ in range(k)] 
    
    # -- calculate average distances -------------------------------------------    
    for i in range(n):
        avg_distance[assignment[i]]+=ml_utils.euclidean_distance(
            centroids[assignment[i]],population[i])        
        cluster_size[assignment[i]]+=1
    
    for i in range(k): avg_distance[i]/=cluster_size[i]
    
    # -- calculate distances between individuals and clusters ------------------    
    distance_cluster=[[0 for _ in range(k)] for _ in range(k)]
    for i in range(k-1):
        for j in range(i+1,k):
            tmp=ml_utils.euclidean_distance(centroids[i],centroids[j])
            distance_cluster[i][j]=tmp
            distance_cluster[j][i]=tmp
    
    
    # -- max values ------------------------------------------------------------
    tmp=0.0
    for i in range(k):
        maxVal=0.0
        for j in range(k):
            if i==j: continue
            tmp=(avg_distance[i]+avg_distance[j])/(distance_cluster[i][j])
            if tmp>maxVal: maxVal=tmp
        
        ret+=maxVal
    
    
    ret/=k 
    return ret



"""
The evaluation function calculates the cuadratic sum of the distances 
of each individual with its cluster centroid. Euclidean distance.

Args:
    population (float[][]) : Population with all the individuals.
    assignment (int[])     : Assignment of categories for each individual.
    centroids (float[][])  : Centroids of each cluster.
"""
def evaluation(population, assignment, centroids):
    n=len(population)    # number of individuals    
    
    ret=0.0 # euclidean distance    
    for i in range(n):
        ret+=ml_utils.euclidean_distance(population[i],
                                         centroids[assignment[i]])
            
    return ret    


"""
Execute the algorithm.

Args:
    population (float[][]) : Population with all the individuals.
    k (int)                : Number of clusters.
    func (function)        : Euclidean or Manhattan distance.
"""
def execute_algorithm(population, k, func):
    km=KMeans(k, population, func, False)

    time_start=time.time()
    
    assignment, centroids=km.execute()

    time_end=time.time()

    eval=evaluation(population, assignment, centroids)    

    print('Obtained evaluation: {}'.format(eval))
    print('Execution time: {}\n'.format(time_end-time_start))

    plot2D(population,assignment,k)

    return assignment


"""
Execute a depth search with different numbers of K.

Args:
    population (float[][]) : Population with all the individuals.
    max_clust (int)        : Maximum number of clusters.
    times (int)            : Number of times executed each K.     
    func (function)        : Euclidean or Manhattan distance.
"""
def execute_search(population, max_clust, times, func):    
    
    fits=[]
    bests=[]
    bests_centroids=[]

    total_time_s = time.time()
    
    # -- different number of Ks ------------------------------------------------
    for k in range(1,max_clust+1):
        ret=float("inf")

        time_start=time.time()

        for _ in range(times):
            km=KMeans(k, population, func, False)
            assignment,centroids=km.execute()
            tmp=evaluation(population, assignment, centroids)
            
            if tmp<ret:
                ret=tmp
                best=assignment
                best_centroid=centroids

        time_end=time.time()

        print("K: {}\tBest eval: {}\tExecution time: {}".
              format(k,ret,(time_end-time_start)))        
        
        fits.append(ret)
        bests.append(best)
        bests_centroids.append(best_centroid)

    total_time_e=time.time()
    print("Total execution time: {}".format(total_time_e-total_time_s))

    
    # -- GUI -------------------------------------------------------------------    
    DBs=[davies_bouldin(population, i, bests[i-1], bests_centroids[i-1]) 
         for i in range(2,k+1)]    
    dbMejor=calculate_best_DB(DBs)
    
    GUI(max_clust, dbMejor+1, fits,DBs,population, bests[dbMejor+1]) 
 
 
def main():
    
    # -- Init population -------------------------------------------------------    
    filename='100_1_2D'
    population=ml_utils.read_population(filename, 'clusters', 2)    
    
    k=4

    # -- Individual distance ---------------------------------------------------    
    dist_ind=0
    dist_ind_name=['Manhattan', 'Euclidean']
    func=ml_utils.euclidean_distance
    if dist_ind==0: func=ml_utils.manhattan_distance

    print('\nDistance: {}\n'.format(dist_ind_name[dist_ind]))

    
    #execute_algorithm(population, k, func)    
    execute_search(population, k, 8, func)

    

main()
