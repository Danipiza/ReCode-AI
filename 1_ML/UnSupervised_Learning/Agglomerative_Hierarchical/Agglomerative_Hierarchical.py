import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import random
import os
import math
import sys
import time

# make copies of lists
import copy

utils_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(utils_dir)

import ml_utils # type: ignore

# EXECUTE
# py Aglomerative.py


"""
Agglomerative Hierarchical class.

Args:
    population (float[][]) : Population with all the individuals.
    C (int)                : Final number of clusters. 
                                used to finalize the execution.
    dist_cluster (int)     : Type of distance used between clusters.
    func (function)        : Euclidean or Manhattan distance.
"""
class AgglomerativeHierarchical:
    
    def __init__(self, population, C, dist_cluster, func):
        self.population=population        
        self.d=len(population[0])         
        self.n=len(population)            
        self.C=C                          
        self.dist_cluster=dist_cluster            
        self.func=func
        
    """Execute function"""
    def execute(self):
        if self.dist_cluster==0: return self.centroid_execution()
        else:return self.execute_link()
        
    
    """
    Execute centroid cluster distance.

    Return:
        assignment (int[][])  : Assignment of categories for each individual
                            in the self.C different stored number of clusters.
        centroids (float[][]) : Centroids for each individual
                            in the self.C different stored number of clusters.
    """
    def centroid_execution(self):

        # -- 1st PHASE: Init Matrix --------------------------------------------
        M=[[0 for _ in range(self.n)] for _ in range(self.n-1)]        
        for i in range(self.n-1):                       
            for j in range(i+1,self.n):
                M[i][j]=self.func(self.population[i],self.population[j])
       

        # at the beginning, all the individuals are clusters
        clusters=[[i] for i in range(self.n)]        
        # centroids of each cluster
        centroid_clusters=[x for x in self.population]        
        
        # -- return variables --------------------------------------------------
        assignment=[[] for _ in range(self.C)]
        centroids =[[] for _ in range(self.C)]
        
        
        
        # -- 2nd PHASE: Main loop ----------------------------------------------
        # repeat the algorithm until there is only 'self.C' clusters
        for k in range(self.C,self.n):  

            # -- search of the 2 closer clusters -------------------------------       
            c1,c2=None,None
            distMin=float("inf")            
            for i in range(len(M)):
                for j in range(i+1,len(M[0])):
                    if distMin>M[i][j]: 
                        distMin=M[i][j]
                        c1=i
                        c2=j            
            
            # -- Join the 2 clusters -------------------------------------------  
            # add all the individuals to the lower index cluster
            for x in clusters[c2]: clusters[c1].append(x)                
            clusters.pop(c2) # delete the greater index cluster
           
            # -- update centroid -----------------------------------------------          
            tmp=[0.0 for _ in range(self.d)]
            # sum the individuals of the cluster
            for x in clusters[c1]:
                for y in range(self.d):
                    tmp[y]+=self.population[x][y]
            
            # divide each dimension
            for x in range(self.d): tmp[x]/=len(clusters[c1])
            centroid_clusters[c1]=tmp
            # delete de greatest cluster
            centroid_clusters.pop(c2)
           

            # -- delete row ----------------------------------------------------
            # last cluster it is not necessary to delete the row  
            if c2!=len(M): M.pop(c2)
            # -- delete column --------------------------------------------------
            for row in M: del row[c2]
            
            
            # -- update column -------------------------------------------------
            i=0
            while i!=c1:
                M[i][c1]=self.func(centroid_clusters[c1],centroid_clusters[i])                
                i+=1

            
            # -- update row ----------------------------------------------------
            for i in range(c1+1,len(M[c1])):
                M[c1][i]=self.func(centroid_clusters[c1],centroid_clusters[i])

            
            # TODO
            # -- GUI parameters  -----------------------------------------------       
            if self.n-k-1<self.C: 
                assignment[self.n-k-1]=copy.deepcopy(clusters)
                centroids [self.n-k-1]=copy.deepcopy(centroid_clusters)

            
        return assignment, centroids


    """
    Execute link cluster distance. 
        - self.dist_cluster = 1: simple 
        - self.dist_cluster = 2: full
    
    Return:
        assignment (int[][])  : Assignment of categories for each individual
                            in the self.C different stored number of clusters.
        centroids (float[][]) : Centroids for each individual
                            in the self.C different stored number of clusters.
    """
    def execute_link(self):
        
        # -- 1st PHASE: Init Matrix --------------------------------------------        
        M=[[0 for _ in range(self.n)] for _ in range(self.n-1)]               
        for i in range(self.n-1):                       
            for j in range(i+1,self.n):
                M[i][j]=self.func(self.population[i],self.population[j])
           
        # at the beginning, all the individuals are clusters
        clusters=[[i] for i in range(self.n)]        
        # centroids of each cluster
        centroid_clusters=[[x] for x in self.population]
        
        # -- return variables --------------------------------------------------
        assignment=[[] for _ in range(self.C)]
        centroids =[[] for _ in range(self.C)]

        
        # -- 2nd PHASE: Main loop ----------------------------------------------
        # repeat the algorithm until there is only 'self.C' clusters
        for k in range(self.C,self.n):  
            
            
            # -- search of the 2 closer clusters -------------------------------            
            c1,c2=None,None
            distMin=float("inf")            
            for i in range(len(M)):
                for j in range(i+1,len(M[0])):
                    if distMin>M[i][j]: 
                        distMin=M[i][j]
                        c1=i
                        c2=j            
            
            
            # -- Join the 2 clusters -------------------------------------------            
            # add all the individuals to the lower index cluster
            for x in clusters[c2]: clusters[c1].append(x)                
            clusters.pop(c2) # delete the greater index cluster
           
            # -- update centroid -----------------------------------------------            
            for x in centroid_clusters[c2]:
                centroid_clusters[c1].append(x)  

            centroid_clusters.pop(c2)
            
            # -- delete row ----------------------------------------------------
            # last cluster it is not necessary to delete the row            
            if c2!=len(M): M.pop(c2)                      
            # -- delete column --------------------------------------------------
            for row in M: del row[c2]
            
            
            # -- parameters ----------------------------------------------------
            def compare_simple(a, b): return a>b
            def compare_full  (a, b): return a<b

            compare=compare_simple
            limit=float("inf")
            if self.dist_cluster==2:
                compare=compare_full
                limit*=-1

            
            # -- update column -------------------------------------------------
            i=0
            c1N=len(centroid_clusters[c1])            
            while i!=c1:
                new_dist=limit
                tmp_dist=limit

                c2N=len(centroid_clusters[i])
                for a in range(c1N):     # iterate through all 'c1' individuals
                    for b in range(c2N): # iterate through all 'c2' individuals
                        tmp_dist=self.func(centroid_clusters[c1][a],
                                           centroid_clusters[i][b])
                        
                        # lower or greater distance
                        if compare(new_dist,tmp_dist): new_dist=tmp_dist 

                M[i][c1]=new_dist                
                i+=1

        
            # -- update row ----------------------------------------------------        
            for i in range(c1+1,len(M[c1])):
                new_dist=limit
                tmp_dist=limit
                c2N=len(centroid_clusters[i])

                for a in range(c1N):     # iterate through all 'c1' individuals
                    for b in range(c2N): # iterate through all 'c2' individuals
                        tmp_dist=self.func(centroid_clusters[c1][a],
                                           centroid_clusters[i][b])

                        # lower or greater distance
                        if compare(new_dist,tmp_dist): new_dist=tmp_dist 

                M[c1][i]=new_dist

               
            
            # TODO
            # -- GUI parameters  -----------------------------------------------            
            if self.n-k-1<self.C: 
                assignment[self.n-k-1]=copy.deepcopy(clusters)
                centroids [self.n-k-1]=copy.deepcopy(centroid_clusters)

        return assignment, centroids




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
Execute the algorithm. With the given parameters.

Args:
    population (float[][]) : Population with all the individuals.
    C (int)                : Final number of clusters. 
                                used to finalize the execution.
    dist_cluster (int)     : Type of distance used between clusters.
    func (function)        : Euclidean or Manhattan distance.
"""
def execute(population, C, dist_cluster, func):        
    n=len(population)    # number of individuals
    d=len(population[0]) # number of dimensions (variables of each individual)
    
    timeStart=time.time()

    AH=AgglomerativeHierarchical(population,C,dist_cluster,func)
    asignaciones, centroids=AH.execute()
    print(asignaciones)

    
    
    # -- GUI -------------------------------------------------------------------
    # -- Individuals assignment ------------------------------------------------
    
    assignment=[[-1 for _ in range(n)] for _ in range(C)]
    for num_clust in range(C):
        for i in range(num_clust+1):
            for j in asignaciones[num_clust][i]:
                assignment[num_clust][j]=i

    timeEnd=time.time()
    print("Execution time: {}\n".format(timeEnd-timeStart))
    
    # -- GUI -------------------------------------------------------------------
    if dist_cluster!=0: # links distances        
        centroids_tmp=[] 

        for num_clust in range(C):
            tmpClust=[]
            for j in range(num_clust+1):
                m=len(centroids[num_clust][j])
                tmp=[0 for _ in range(d)]
                for x in centroids[num_clust][j]:
                    for a in range(d):
                        tmp[a]+=x[a]
                for a in range(d):
                    tmp[a]/=m
                tmpClust.append(tmp)                      

            centroids_tmp.append(tmpClust)    
        
        centroids=centroids_tmp  

    # fitness evaluation of each asignment.    
    fits=[evaluation(population, assignment[i],centroids[i]) 
          for i in range(C)]
    # Davies Bouldin index
    DBs=[davies_bouldin(population, i, assignment[i-1], centroids[i-1]) 
         for i in range(2,C+1)]  
    
    # best DB index.
    best_DB=calculate_best_DB(DBs)
    
    GUI(C, best_DB+1, fits,DBs,population, assignment[best_DB+1])

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



def main():

    # -- Init population -------------------------------------------------------    
    filename='100_1_2D'
    C=7 # minimum number of clusters
    population=ml_utils.read_population(filename, 'clusters', 2)    
    
    # -- Individual distance ---------------------------------------------------    
    dist_ind=0
    dist_ind_name=["Manhattan", "Euclidean"]
    func=ml_utils.euclidean_distance
    if dist_ind==0: func=ml_utils.manhattan_distance
    
    # -- Cluster distance ------------------------------------------------------
    dist_cluster=0
    dist_cluster_name=['Centroid','Simple Link','Full Link']

    
    print('Cluster distance: {}\tIndividual distance: {}\n'
          .format(dist_cluster_name[dist_cluster], dist_ind_name[dist_ind]))
    
    
    # -- Execute ---------------------------------------------------------------
    
    execute(population, C, dist_cluster, func)

    

main()


