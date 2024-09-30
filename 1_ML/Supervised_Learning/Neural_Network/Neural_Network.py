import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import time
import random
import sys
import os
import math

utils_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(utils_dir)

import ml_utils # type: ignore

# EJECUTAR
# py RN_2_1.py

"""
GUI: With two plots. 
 - Errors
 - Best num_epochs prediction.

Args:
    lr (float)           : Learning rate.
    errors (float[])     : Errors for each lr different execution.
    num_epochs (float[]) : Best num_epoch for each lr different execution
"""
def GUI(lrs, errors, num_epochs):    
    # porcentage
    x=[lr*100 for lr in lrs]
    
    # figures in a grid
    fig =plt.figure(figsize=(10, 6))
    gs  =GridSpec(2, 1, figure=fig)

    
    # --- 1st Plot -------------------------------------------------------------    
    ax1=fig.add_subplot(gs[0, 0])
    ax1.plot(x, errors, color='b', linestyle='-')    
    ax1.set_xlabel('Learning Rate (%)')
    ax1.set_ylabel('Errors')
    ax1.set_title('Bests')
    ax1.grid(True)

    # --- 2nd Plot -------------------------------------------------------------
    ax2=fig.add_subplot(gs[1, 0])
    ax2.plot(x, num_epochs, color='b', linestyle='-')     
    ax2.set_xlabel('Learning Rate (%)')
    ax2.set_ylabel('Num. Epochs')
    ax2.set_title('Bests')
    ax2.grid(True)
       
    
    plt.tight_layout()   
    plt.show() 


"""
Calculate the minimum value of an array.

Args:
    fila (float[]) : Array.

Return:
    ret (float) : Minimum value.
    index (int) : Index of the minimum value.
"""
def minimum_val(fila):
    ret=float("inf")
    index=-1

    n=len(fila)

    for i in range(n):
        if ret>fila[i]: 
            ret=fila[i]
            index=i

    return ret, index

"""
Normalize data.

Args:
    data (float) : Data to normalize.
    m (float)    : Minimum value.
    M (float)    : Maximum value.

Return:
    ret (float) : Normalizated data.
"""
def normalize_data(data,m,M):
    return (data-m)/(M-m)

"""
Denormalize data.

Args:
    data (float) : Data to denormalize.
    m (float)    : Minimum value.
    M (float)    : Maximum value.

Return:
    ret (float) : Denormalizated data.
"""
def desnormalizar_dato(data,m,M):
    return data*(M-m)+m


"""Activation function:"""
def sigmoide(x): return 1/(1+math.exp(-x))

"""Activation function for the trainning"""
def sigmoide_derivado(x): return x*(1-x)


"""
Neural Network class.

Args:
    input_size (int)      : Input layer size.
    hidden_size (int[])   : Hidden layers sizes.
    output_size (int)     : Output layer size.
    weights (float[][][]) : Neural Network model.
    lr (float)            : Learning rate.
"""
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, 
                 weights, lr):
        self.input_size  =input_size
        self.hidden_size =hidden_size
        self.output_size =output_size
        
        self.lr=lr        
        self.layers=[input_size]+hidden_size+[output_size]
        
        # random init.
        if weights==None: 
            self.weights=[]

            for i in range(len(self.layers)-1):
                layer_weights=[
                    [random.uniform(-1, 1) for _ in range(self.layers[i + 1])] 
                    for _ in range(self.layers[i])]
                
                self.weights.append(layer_weights)
            
            #print(self.pesos)

        # load model
        else: self.weights=weights
        
        

    
    """
    Forward propagation.

    Args:
        x (float[]) : Individual

    Return:
        prediction (float) : BMI model prediction.
    """
    def forward(self, x):
        self.output=[x]

        # iterate through the layers of the network.
        for i in range(len(self.layers)-1):
            inputs_layer=self.output[-1]
            outputs_layer=[0 for _ in range(self.layers[i+1])]

            # iterates through all the neurons of the next layer
            for j in range(self.layers[i+1]):    
                sum_v=0

                # sums the values of the input with 
                #   the weights of the next layer                
                for k in range(self.layers[i]):            
                    sum_v+=inputs_layer[k]*self.weights[i][k][j]
                
                outputs_layer[j]=sigmoide(sum_v) # activation function
            
            self.output.append(outputs_layer)
        
        # return the prediction
        # the last layer (output) only has one neuron
        return self.output[-1] 

    """
    Backpropagation method.

    Args:
        x (float[])   : Individual
        label (float) : BMI value of the individual.
    """
    def backpropagation(self, x, label):
        self.forward(x)

        errors=[]
        for i in range(self.output_size):
            errors.append((label[i]-self.output[-1][i])*\
                          sigmoide_derivado(self.output[-1][i]))
                        
        # iterate through the layers of the network. inverse path
        for i in range(len(self.layers)-2,-1,-1):
            new_errors=[0 for _ in range(self.layers[i])]
            
            # iterates through all the neurons of the actual layer
            for j in range(self.layers[i]):
                sum_v=0

                # sums the values of the input with 
                #   the weights of the next layer 
                #   (without the inverse path, the right layer)                  
                for k in range(self.layers[i+1]):            
                    sum_v+=errors[k]*self.weights[i][j][k]
                
                # acitvation function
                new_errors[j]=sum_v*sigmoide_derivado(self.output[i][j])

                # Update the weights of the network
                for k in range(self.layers[i+1]):
                    self.weights[i][j][k]+=self.lr*errors[k]*self.output[i][j]

            errors=new_errors


"""
Execute the algorithm.

Args:
    dataset (float[][])     : Dataset for the training session.
    eva_dataset (float[][]) : Dataset for the evaluation method.
    input_size (int)        : Size of the input layer.
    hidden_size (int[])     : Sizes of the hidden layer. 
    output_size (int)       : Size of the output layer.
    model (float[][][])     : Neural Network model.
"""
def execute(dataset, eval_dataset, num_epochs, lr,
            input_size, hidden_size, output_size, model):
    timeStart=time.time()
    
    nn=NeuralNetwork(input_size,hidden_size,output_size, model, lr)    
    error=trainning_method(dataset, eval_dataset, nn, num_epochs)    

    timeEnd=time.time()    
    print('Final error: {}\nExecution time: {}s\n'.format(error[-1], timeEnd-timeStart))
    

"""
Trainning method of the model.

Args:
    dataset (float[][])     : Dataset for the training session.
    eva_dataset (float[][]) : Dataset for the evaluation method.
    nn (model)              : Neural Network model
    num_epochs (int)        : Number of epochs.
    PRINT (bool)            : Boolean to print the epoch information.
"""
def trainning_method(dataset, eval_dataset,
                      nn, num_epochs,
                      PRINT=False):
    ret=[]

    for epoch in range(num_epochs):
        for data in dataset:                    
            input=data[:2]
            label=[data[2]]
            nn.backpropagation(input,label)
        
        error=0
        for data in eval_dataset:
            prediction=nn.forward(data[0:2])      
            error+=abs(prediction[0]-data[2]) 
        
        ret.append(error)
        if PRINT: print("Epoch {} - Total error = {}".format(epoch, error))

    return ret

"""
Execute a search of the optimal learning rate.

Args:
    dataset (float[][])     : Dataset for the training session.
    eva_dataset (float[][]) : Dataset for the evaluation method.
    input_size (int)        : Size of the input layer.
    hidden_size (int[])     : Sizes of the hidden layer. 
    output_size (int)       : Size of the output layer.
    model (float[][][])     : Neural Network model.
"""
def execute_search(dataset, eval_dataset, num_epochs,
                   input_size, hidden_size, output_size, model):
    print("Hidden layer sizes: {}\tNumber of epochs: {}\n".format(hidden_size, num_epochs))
    timeStart=time.time()
    
    learning_rates=[0.01*i for i in range(1,21)]          
    errors=[]
    epochs=[]
    
    for lr in learning_rates:        

        nn=NeuralNetwork(input_size,hidden_size,output_size, model, lr)        
        lr_error=trainning_method(dataset, eval_dataset, 
                                  nn, num_epochs)       

        err, rep=minimum_val(lr_error)
        errors.append(err) 
        epochs.append(rep)

    timeEnd=time.time()
    print("Execution time: {}s\n".format(timeEnd-timeStart))

    GUI(learning_rates, errors, epochs)

def main():
    
    # --- Data -----------------------------------------------------------------
    # (height, weight, BMI)
    dataset=ml_utils.read_population('population_80','neural_network',3)      
    
    # --- Normalize data -------------------------------------------------------
    

    heights =[data[0] for data in dataset]
    weights =[data[1] for data in dataset]
    bmis    =[data[2] for data in dataset]

    max_height =min(heights)
    min_height =max(heights)
    min_weight =min(weights)
    max_weight =max(weights)
    min_bmi    =min(bmis)
    max_bmi    =max(bmis)

    dataset=[
        [normalize_data(data[0], min_height, max_height), 
         normalize_data(data[1], min_weight, max_weight), 
         normalize_data(data[2], min_bmi, max_bmi)
        ] for data in dataset]
    
    eval_dataset=dataset[:10]
    
    
    
    # --- Neural Network -------------------------------------------------------
    # Input Layer: height and weight
    input_size =2               
    # Hidden Layer. size for each layer
    hidden_size=[10 for _ in range(1)] 
    # Output Layer: bmi
    output_size=1       
    
    # weights of the neural network. 
    #   none      = random init. 
    #   otherwise = load model.
    model=None   
    num_epochs=10 
    lr=0.001

    # --- Loop -----------------------------------------------------------------

    """execute(dataset, eval_dataset, num_epochs, lr,
            input_size, hidden_size, output_size, model)"""
    execute_search(dataset, eval_dataset, num_epochs,
                   input_size, hidden_size, output_size, model)
    
 

main()
