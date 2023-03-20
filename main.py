import pandas as pd
import sqlalchemy as db
import mysql.connector


def csv_to_sql():
    # Load the CSV file into a pandas dataframe
    dfCsv = pd.read_csv(filepath_or_buffer='Hotel_Reviews.csv', engine='python', on_bad_lines='skip')
    # drop unwanted columns
    dfCsv.drop(['Hotel_Address','Additional_Number_of_Scoring', 'Review_Date', 'Average_Score',
             'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews',
             'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given',
             'Reviewer_Score', 'Tags', 'days_since_review', 'lat', 'lng'], axis=1, inplace=True)
    # create db first in MySQL
    engine = db.create_engine('mysql+mysqlconnector://root:Rg123456@localhost/HotelReviews')
    conn = engine.connect()
    # Create table object
    dfCsv.to_sql(name='reviews', con=engine, if_exists='replace', chunksize=1000)
    # Remove dataframe
    del dfCsv
csv_to_sql()

def create_stored_procedure():
    # Connect to the database
    cnx = mysql.connector.connect(user='root', password='Rg123456', host='localhost', database='HotelReviews')
    cursor = cnx.cursor()
    # Check if the stored procedure exists
    check_sp_query = "SHOW PROCEDURE STATUS WHERE Name = 'get_model_data'"
    cursor.execute(check_sp_query)
    result = cursor.fetchone()
    # If the stored procedure exists, drop it
    if result:
        drop_sp_query = "DROP PROCEDURE IF EXISTS `get_data`"
        cursor.execute(drop_sp_query)
    # Create the stored procedure
    create_sp_query = """
    CREATE PROCEDURE `get_data`(IN limitAmount int)
    BEGIN
        SELECT Hotel_Name, Reviewer_Nationality, Negative_Review, Positive_Review
        FROM reviews
        LIMIT limitAmount;
    END
    """
    cursor.execute(create_sp_query)
    cursor.close()
    cnx.close()
create_stored_procedure()

# Call the stored procedure

