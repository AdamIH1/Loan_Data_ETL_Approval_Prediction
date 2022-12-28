# imports 
import os 
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
 

# paths 
base_path = os.path.abspath('kaggle_api' + "/../../../")
destination_path = f"{base_path}/loan_approval_data"

def create_folder_if_not_exists():
    """
    Create a new folder if it doesn't exists
    """ 
    try:
        os.makedirs(destination_path)
        print(f"File {destination_path.split('/')[-1]} created")
    except: 
        print(f"File {destination_path.split('/')[-1]} already exists or error")

def api_request():
        
    # initilalize api and authenticate

    api = KaggleApi()
    api.authenticate()
    
    # downloading datasets 
    try:
        api.dataset_download_file('granjithkumar/loan-approval-data-set',
                        file_name='Loan_Train.csv',
                        path='/Users/adamisaiahhansen/Downloads/projects/loan_approval_data')
        print(f"Data has been downloaded to {destination_path.split('/')[-1]}")
    except: 
        print('Error with api dataset download')


def main(): 
    create_folder_if_not_exists()
    api_request()
    print('extract is done')