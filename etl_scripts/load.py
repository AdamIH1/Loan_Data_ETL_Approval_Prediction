# imports 
import mysql.connector # for load data infile option  
from sqlalchemy import create_engine # pandas to_sql option
from sqlalchemy.pool import NullPool
import pandas as pd 

from transform import newest_file # gets newest file form dir 

# to_sql 
engine = create_engine('mysql+mysqlconnector://YOURROOT:YOURPASSWORD@YOURHOST/loan_db', poolclass = NullPool)

# load data infile 
mydb = mysql.connector.connect(
    host="YOURHOST",
    user="YOURROOT",
    password="YOURPASSWORD",
    database="loan_db", 
    allow_local_infile=1
)

cursor = mydb.cursor()

# paths 
dir_path = '/Users/adamisaiahhansen/Downloads/projects/loan_approval_data'
pattern = '*.csv'

db = 'loan_db'
db_table = 'loan_approval'


def load_file():
    """
    adds clean data from transformation to loan_db.loan_approval
    this function show two ways of adding data 
        1. using pandas to_csv, reading chunks is not nessacary as 
            the data small 
        2. Load data infile which is faster but using local 
            has potential security issues
    
    """
    
    new_file = newest_file(dir_path, pattern)
    
    try: 
        for data in pd.read_csv(f"{dir_path + '/' + new_file}",chunksize=100):
            data.to_sql(name = f"{db_table}", con = engine
                    ,if_exists= 'append', index = False , method='multi')
    
    except: 
        print('error in load')
        
    engine.dispose()
    
#     try:
#         query = (f"LOAD DATA LOCAL INFILE '{dir_path + '/' + new_file}'\n"
#                 f"INTO TABLE {db}.{db_table}\n"
#                 "FIELDS TERMINATED BY','\n"
#                 "LINES TERMINATED BY '\n'"
#                 "IGNORE 1 ROWS"
#                  )
#         cursor.execute(query)
#         mydb.commit()
#         cursor.close()
#         mydb.close()
        
#         print(f"{dir_path.split('/')[-1]} has been added to {db_table} table")
        
#     except: 
#         print(f"error with {dir_path.split('/')[-1]} or {db_table} table")

def main(): 
    try:
        load_file()
        print('load is done')
    except: 
        print('Error in load')