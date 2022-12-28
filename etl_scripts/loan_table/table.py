import mysql.connector


mydb = mysql.connector.connect(
  host="YOURHOST",
  user="YOURROOT",
  password="YOURPASSWORD",
  database="loan_db", 
  database="loan_db"
)


q = '''CREATE TABLE loan_approval( 
            Loan_ID VARCHAR(255) DEFAULT NULL
            , Gender VARCHAR(255) DEFAULT NULL
            , Married VARCHAR(255) DEFAULT NULL
            , Dependents VARCHAR(255) DEFAULT NULL 
            , Education VARCHAR(255) DEFAULT NULL
            , Self_Employed VARCHAR(255) DEFAULT NULL
            , ApplicantIncome VARCHAR(255) DEFAULT NULL
            , CoapplicantIncome VARCHAR(255) DEFAULT NULL
            , LoanAmount VARCHAR(255) DEFAULT NULL
            , Loan_Aount_Term INT DEFAULT NULL
            , Credit_History INT DEFAULT NULL 
            , Property_Area VARCHAR(255) DEFAULT NULL 
            , Loan_Status VARCHAR(255)) DEFAULT NULL 
            '''

def loan_table_query(): 
    cursor = mydb.cursor()
    cursor.execute(q)
    cursor.close()
    mydb.close()

def main(): 
    print('started creating loan approval table')
    loan_table_query()
    print('finished loan approval table')