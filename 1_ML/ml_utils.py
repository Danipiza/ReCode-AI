import math
import os


"""
K-Nearest Neighbors (KNN) algorithm.

Individual distances:
 - Manhattan
 - Euclidea

Once the algorithm finalize, an interface shows the assignment of:
    Initial population (at the left) 
    Final population (at the right)
"""

"""
Calculates de Euclidean distance of 
    two points in a 'd' dimensional space.

Args:
    a (float[]) : Point in a d-dimensional space.
    b (float[]) : Point in a d-dimensional space.
"""
def euclidean_distance(a, b):
    ret=0
    d=len(a)
    for i in range(d):
        ret+=(a[i]-b[i])**2    

    return math.sqrt(ret)

"""
Calculates de Manhattan distance of 
    two points in a 'd' dimensional space.

Args:
    a (float[]) : Point in a d-dimensional space.
    b (float[]) : Point in a d-dimensional space.
"""
def manhattan_distance(a, b):
    ret=0
    d=len(a)
    for i in range(d):
        ret+=abs(a[i]-b[i])

    return math.sqrt(ret)


"""
Read a population of individuals with 'd' variables.

The format of the file has to be as follows:
[_,..,_], [_,..,_], ... ,[_,..,_]
(All in 1 line.)

Args:
    file_name (string) : Name of the file that is going to be readed.

Return:
    ret (float[][]) : Individuals of the population.
"""
def read_population(file_name, algorithm, d):    
    ret   =[]            # return list
    point =[]            # individual    
    
    # get the root directory of the proyect
    dir=os.getcwd()       
    while(dir[-9:]!='ReCode-AI'):
        dir=os.path.dirname(dir)

    # name of the file
    if file_name==None: file_name=input('Introduce a file name: ')    
    path=os.path.join(dir,'.Others','files',algorithm, file_name+'.txt')

    # read the line    
    try:
        with open(path, 'r') as file:
            data=file.read()
    except FileNotFoundError:
        print("The file '{}' doesnt exits.".format(file_name+".txt"))
    

    # removes '[', ']'. And divides the file by ','
    data=data.replace('[', '').replace(']', '').split(', ') 
    
    # store the individuals in the return list     
    for i in range(0, len(data), d):
        point=[]
        # read and store all the variables of the individual
        for j in range(d):
            point.append(float(data[i+j]))
        
        ret.append(point)
              
    
    return ret


"""
Read an assignment of individuals.

The format of the file has to be as follows:
_ _ ... _
(All in 1 line.)

Args:
    file_name (string) : Name of the file that is going to be readed.

Return:
    ret (int[]) : Categories of each individual.
"""
def read_assignment(file_name):
    ret=[] # return list

    # get the root directory of the proyect
    dir=os.getcwd()       
    while(dir[-9:]!='ReCode-AI'):
        dir=os.path.dirname(dir)

    # name of the file
    if file_name==None: file_name=input('Introduce a file name: ')    
    path=os.path.join(dir,'.Others','files','clusters','assignment', file_name+'.txt')
    
    # read the line
    try:
        with open(path, 'r') as file:
            data=file.read()
    except FileNotFoundError:
        print("The file '{}' doesnt exits.".format(file_name+".txt"))

    data=data.split(' ')                               
    for num in data:
        ret.append(int(num))
    
    
    
    return ret
