import csv

"""
Read Labels from a CSV file.

Return the possible labels in the dataset csv file.

Args:
    file_path (string) : Path to the file that is going to be readed.
"""
def read_labels_csv(file_path):
    ret=set() 

    with open(file_path, mode='r', newline='') as file:
        reader=csv.reader(file)
        
        for row in reader:
            if row:  # not empty
                label=row[0].split()[0]
                ret.add(label)  
    
    return ret

def main():
    path       ='data.csv'  
    categories =read_labels_csv(path)

    print(categories)

main()