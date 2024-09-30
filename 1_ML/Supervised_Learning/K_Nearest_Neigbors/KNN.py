import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import os
import sys
import time

import queue

utils_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(utils_dir)

import ml_utils # type: ignore

# EXECUTE
# py KNN.py




"""
PriorityQueue used to optimize the K nearest neighbors calculations.

It is a maximum priority queue to store the nearest neighbors. 
If a new individual is closer than the farest closest neighbor, 
    pops the top and push the new individual in the queue.

Variables:
    priority = distance
    item     = index
"""
class MaxPriorityQueue(queue.PriorityQueue):
    def __init__(self):
        super().__init__()

    def push(self, item, priority):
        super().put((-priority, item))

    def top_distance(self):
        priority, _ =self.queue[0]  

        return -priority
    
    def top_label(self):
        _, item =self.queue[0]  
        
        return item
    
    def pop(self):
        _, item = super().get()
        return item
    
    def size(self):
        return self.qsize()


def GUI(num_clusters, init_population, init_assig, n, 
                  population,assignment, m):    
    
    colors = ['blue', 'red', 'green', 'pink', 'yellow', 'magenta', 'brown', 'darkgreen', 'gray', 'fuchsia',
            'violet', 'salmon', 'darkturquoise', 'forestgreen', 'firebrick', 'darkblue', 'lavender', 'palegoldenrod',
            'navy']
    
    # create the figures and axis
    fig, axs = plt.subplots(1,2, figsize=(12, 6))    
    
    x1=[[] for _ in range(num_clusters)]
    y1=[[] for _ in range(num_clusters)]
    
    for i in range(n):        
        x1[init_assig[i]].append(init_population[i][0])
        y1[init_assig[i]].append(init_population[i][1])          

    
    for i in range(num_clusters):
        axs[0].scatter(x1[i], y1[i], color=colors[i])
    
    for i in range(m):
        axs[0].scatter(population[i][0], population[i][1], color="black")
    
    
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('Poblacion Ini')


    for i in range(m):
        x1[assignment[i]].append(population[i][0])
        y1[assignment[i]].append(population[i][1]) 
    

    
    for i in range(num_clusters):
        axs[1].scatter(x1[i], y1[i], color=colors[i])    
    
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Y')
    axs[1].set_title('Poblacion Fin')
    


    plt.tight_layout()
    plt.show()




"""
Classify one individual.

Iterates through the given population and store in a priority queue
    the 'k' nearest neighbors and classify with the assignment
    of the stored individuals.

Args:
    population (float[][]) : Categorized population.
    assignment (int[])     : assignment of the population.
    ind (float[])          : Individual
    num_clusters (int)     : Number of possible clusters.
    k (int)                : Number of neighbors.
    func (function)        : Manhattan or Euclidean distance.

Return:
    ret (int) : Asignated cluster for the individual.
"""
def classify_individual(population, assignment, ind, 
                        num_clusters, k, func):
    
    n=len(population)
    pq=MaxPriorityQueue()

    
    
    # calculate all distance an store the K nearest
    for i in range(n):
        distance=func(ind, population[i])
               
        
        # space in queue?
        if pq.size()<k: pq.push(assignment[i],distance)
        
        # if the actual distance is lower than the greater nearest distance
        # pops the greater and push the actual distance  
        elif pq.top_distance()>distance:            
            pq.pop()
            pq.push(assignment[i],distance)


    # counts the number of neighbors for each cluster
    labels=[0 for _ in range(num_clusters)]    
    for i in range(k):
        labels[pq.pop()]+=1
    
    # classify the individual with the 
    #   cluster with more occurrences
    ret=0
    max_occur=labels[0]
    for i in range(1,num_clusters):
        if max_occur<labels[i]:
            max_occur=labels[i]
            ret=i
               
	
    return ret

"""
Classify a population without updating.

Iterates to classify each individual of the given population.

Args:
    init_population (float[][]) : Categorized population.
    init_assig (int[])           : assignment of the population.
    n (int)                     : Number of individuals in the categorized population
    population (float[][])      : Population to categorize.
    n (int)                     : Number of individuals in the population to categorize
    num_clusters (int)          : Number of possible clusters.
    k (int)                     : Number of neighbors.
    func (function)             : Manhattan or Euclidean distance.
    
Return:
    ret (int) : Asignated cluster for the individual.
"""
def execute_no_update(init_population, init_assig, n, 
                      population, m, 
                      num_clusters, k, func):           
    time_start=time.time()

    assignment=[]
    for i in range(m):           
        assignment.append(classify_individual(init_population, init_assig, population[i], 
                                                    num_clusters, k, func))
    
    time_end=time.time()
    print("Execution time: {}\n".format(time_end-time_start))    
    GUI(num_clusters,init_population,init_assig, n, population,assignment,m)

"""
Classify a population. 
The population is updated through the iterations.

Iterates to classify each individual of the given population.

Args:
    init_population (float[][]) : Categorized population.
    init_assig (int[])           : assignment of the population.
    n (int)                     : Number of individuals in the categorized population
    population (float[][])      : Population to categorize.
    n (int)                     : Number of individuals in the population to categorize
    num_clusters (int)          : Number of possible clusters.
    k (int)                     : Number of neighbors.
    func (function)             : Manhattan or Euclidean distance.
    
Return:
    ret (int) : Asignated cluster for the individual.
"""
def execute_update(init_population, init_assig, n, 
                   population, m, 
                   num_clusters, k, func):    
    time_start=time.time()

    first_population=[x for x in init_population]
    first_asig=[x for x in init_assig]


    assignment=[]    
    for x in range(m):                          
        pq=MaxPriorityQueue()        

        
        # calculate all distance an store the K nearest
        for i in range(n):
            distance=func(init_population[i],population[x])  
            
            # space in queue?
            if pq.size()<k: pq.push(init_assig[i],distance)
            
            # if the actual distance is lower than the greater nearest distance
            # pops the greater and push the actual distance
            elif pq.top_distance()>distance:            
                pq.pop()
                pq.push(init_assig[i],distance)

        # counts the number of neighbors for each cluster
        labels=[0 for _ in range(num_clusters)]    
        for i in range(k):
            labels[pq.pop()]+=1
        
        # classify the individual with the 
        #   cluster with more occurrences
        ret=0
        max_occur=labels[0]
        for i in range(1,num_clusters):
            if max_occur<labels[i]:
                max_occur=labels[i]
                ret=i                
        
        # add the new assignment
        assignment.append(ret)
        init_assig.append(ret)

        # update the population
        init_population.append(population[x])        
        n+=1
    
    
    time_end=time.time()
    print("Execution time: {}\n".format(time_end-time_start))

    GUI(num_clusters,first_population,first_asig,
        n-m, population, assignment, m)




def main():
    # 100_1_2D 7
    init_population=ml_utils.read_population("1000_1_2D",'clusters',2)
    init_assig=ml_utils.read_assignment("1000_1_2D") 
    n=len(init_population)    
    
    population=ml_utils.read_population("100000_2D",'clusters',2)
    population=population[0:1000]
    m=len(population)
    
    num_clusters=4
    k=10
    distance=1

    distance_name=["Manhattan", "Euclidean"]
    func=ml_utils.euclidean_distance
    if distance==0: func=ml_utils.manhattan_distance
    actualiza=1
    
    print("Init population size: {}\tPopulation size: {}\tk= {}\tDistance {}".format(n, m, k,
            distance_name[distance]))

    
    if actualiza==0:
        execute_no_update(init_population, init_assig, n, 
                          population, m, 
                          num_clusters, k, func)
    else:
        execute_update(init_population, init_assig, n, 
                       population, m, 
                       num_clusters, k, func)

    
    



main()