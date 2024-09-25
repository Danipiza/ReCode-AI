import os
import matplotlib.pyplot as plt

import re

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

"""
Store values (in 'dst_path') getted using the regular expresion, from 'src_path'.

Args:
    src_path (string)                    : Origin path. 
    dst_path (string)                    : Destination path.
    accuracy_patron (regular expression) : Patron.
"""
def process_files(src_path, dst_path, accuracy_patron):
    # check if the destination directory exists
    if not os.path.exists(dst_path): os.makedirs(dst_path)

    count=0 

    # iterate throw the files of the origin directory
    for file_name in os.listdir(src_path):
        file_path=os.path.join(src_path, file_name)
        
        # process .txt files
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            
            with open(file_path, 'r') as fichero_lectura:                
                lines=fichero_lectura.readlines()
            
            accuracies=[]
            for line in lines:
                # patron using the regular expression
                match=re.search(accuracy_patron, line)
                if match: accuracies.append(match.group(1))  

            # open/create the destination file 
            modified_dst_path=os.path.join(dst_path, file_name)
            with open(modified_dst_path, 'w') as file_w:                
                file_w.write(' '.join(accuracies) + '\n')
            
            count+=1

    print('Done!\tFiles modified: {}\n'.format(count))

"""
Get values from the current directory files.
"""
def get_vals():    
    src_path='./'
    dst_path='./modified'    

    # regular expression. find the value of 'Accuracy'
    accuracy_patron=r'Accuracy: (\d+\.\d{4})'

    process_files(src_path, dst_path, accuracy_patron)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

"""
Read all the files from 'path' with the given prefix.

Args:
    path (string)   : Directory.
    prefix (string) : Prefix of the files.
"""

def read_files(path, prefix):
    ret=[]

    valid_files=[]   

    # search of files with the given prefix
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)) and \
            name.startswith(prefix):
            valid_files.append(name)
    
    # accuracies of each functions
    for file in valid_files:
        file_path=os.path.join(path, file)

        with open(file_path, 'r') as f:
            line=f.readline().strip()

            vals=list(map(float, line.split()))
            ret.append(vals)
    
    return ret

"""
Function used to print the functions of the files.
"""
def GUI(data, epochs):
    plt.figure(figsize=(10, 6))
    
    for i, valores in enumerate(data):
        epochs=list(range(1, len(valores) + 1))
        plt.plot(epochs, valores, label=f'File {i+1}')

    plt.xlabel('Num. Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Epoch: {epochs}')
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.show()

def analyze_results():    
    directory = os.getcwd()
    prefix='50_'    
    num_epochs=int(prefix[:-1])  

    datos = read_files(directory, prefix)

    if datos: GUI(datos, num_epochs)
    else: print("No files with the given prefix.")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    """get_vals()"""
    """analyze_results()"""