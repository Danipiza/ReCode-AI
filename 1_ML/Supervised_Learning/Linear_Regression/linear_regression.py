import torch
from torch import nn
from torch.nn import Linear


"""
Function. y = b + w*x

Args:
    x (tensor) : Value.
    w (float)  : Weight.
    b (float)  : Bias.
"""
def func(x, w, b):
    y=w*x+b

    return y


"""
Simple Linear Regression. Using a function.
"""
def using_function():
    
    w=torch.tensor(2.5,  requires_grad=True) # weight
    b=torch.tensor(-1.2, requires_grad=True) # bias
        
    # -- One Individual --------------------------------------------------------
    x=torch.tensor([[2.0]])
    y=func(x, w, b)

    # detach to remove grad_fn
    print('\n1 individual at once:\n{}'.format(y.detach()))
    
    # -- Multiple Individuals --------------------------------------------------
    x=torch.tensor([[float(i)] for i in range(10)])
    y=func(x, w, b)
    
    print('\n10 individuals at once:\n{}'.format(y.detach()))

"""
Simple Linear Regression. Using the default module of Linear from torch.nn.
"""
def using_default_module():
    # init 'w' and 'b' parameters
    torch.manual_seed(1)

    
    # -- Init ------------------------------------------------------------------    
    linear=Linear(in_features=1, out_features=1, bias=True)
    print('Parameters w and b: ', list(linear.parameters()))

    """
    print('weight:\t',linear.weight)
    print('bias:\t',linear.bias.item())
    """
    dic=linear.state_dict()
    print("\nParameters:")
    for x in dic:
        print(x, dic[x])

    # -- One Individual --------------------------------------------------------
    x=torch.tensor([[1.0]])
    y=linear(x)
    
    print('\n1 individual at once:\n{}'.format(y.detach()))

    # -- Multiple Individuals --------------------------------------------------
    x=torch.tensor([[float(i)] for i in range(10)])
    y=linear(x)
    
    print('\n10 individuals at once:\n{}'.format(y.detach()))

"""
Custome Module to use instead of the default from torch.nn
"""
class CustomeModule(nn.Module):    
    
    def __init__(self, input_size, output_size):
        super(CustomeModule, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)
    
    # Prediction function
    def forward(self, x):
        out = self.linear(x)
        return out

"""
Simple Linear Regression. Using a custome module.
"""
def using_custome_module():
    # init 'w' and 'b' parameters
    torch.manual_seed(1)

    # -- Init ------------------------------------------------------------------
    module=CustomeModule(1, 1)
    dic=module.state_dict()
    print("\nParameters:")
    for x in dic:
        print(x, dic[x])
        
    # -- One Individual --------------------------------------------------------    
    x=torch.tensor([[1.0]])
    y=module(x)
    print('\n1 individual at once:\n{}'.format(y.detach()))

    # -- Multiple Individuals --------------------------------------------------
    x=torch.tensor([[float(i)] for i in range(10)])
    y=module(x)
    print('\n10 individuals at once:\n{}'.format(y.detach()))




def main():
    using_function()
    using_default_module()
    using_custome_module()


main()
