# imports 
import os 
import csv 
import pandas as pd
import glob

# paths 
dir_path = '/Users/adamisaiahhansen/Downloads/projects/loan_approval_data'
pattern = '*.csv'
outputfile = 'Clean_Loan_Approval_Data.csv'

def newest_file(path, file_pattern): 
    """
    Returns newest file from a dir
    
    """
    try: 
        files = glob.glob(path + '/' + file_pattern)
        files.sort(key=os.path.getmtime)
        newest_csv = files[-1].split('/')[-1]
        
        return newest_csv
        print('newest file found')
    
    except: 
        print('error while finding newest file, dir may be empty')

def transform_case(string_feature):
    """
    Lowercase string fields
    
    """
    return string_feature.str.lower()
    
    
def transform_data():
    
    new_file = newest_file(dir_path, pattern)
    
    try: 
        for data in enumerate(pd.read_csv(f"{dir_path + '/' + new_file}",chunksize=100)):
            if data[0] == 0:
                data[1]['Gender'] =  transform_case(data[1]['Gender'])
                data[1]['Married'] =  transform_case(data[1]['Married'])
                data[1]['Education'] =  transform_case(data[1]['Education'])
                data[1]['Self_Employed'] =  transform_case(data[1]['Self_Employed'])
                data[1]['Property_Area'] =  transform_case(data[1]['Property_Area'])

                data[1]['Loan_Status'] = data[1]['Loan_Status'].map({'Y':1 ,'N':0})

                data[1].to_csv(f"{dir_path + '/' + outputfile}", index = False, mode = 'a')
            else: 
                data[1]['Gender'] =  transform_case(data[1]['Gender'])
                data[1]['Married'] =  transform_case(data[1]['Married'])
                data[1]['Education'] =  transform_case(data[1]['Education'])
                data[1]['Self_Employed'] =  transform_case(data[1]['Self_Employed'])
                data[1]['Property_Area'] =  transform_case(data[1]['Property_Area'])

                data[1]['Loan_Status'] = data[1]['Loan_Status'].map({'Y':1 ,'N':0})

                data[1].to_csv(f"{dir_path + '/' + outputfile}", mode = 'a', index = False, header = False)
        print(f'Transformation successful, file added to {dir_path}')
    except: 
        print('Error in transformation')

def main(): 
    print('Starting Transformation')
    transform_data()
    print('Transformation Complete')