import pandas as pd
import sqlalchemy as db
from sqlalchemy import text
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
    conn.close()
csv_to_sql()

def create_stored_procedure():
    # Connect to the database
    cnx = mysql.connector.connect(user='root', password='Rg123456', host='localhost', database='HotelReviews')
    cursor = cnx.cursor()
    # Check if the stored procedure exists
    check_sp_query = "SHOW PROCEDURE STATUS WHERE Name = 'get_data'"
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

def call_stored_procedure(limit):
    # Connect to the database
    cnx = mysql.connector.connect(user='root', password='Rg123456', host='localhost', database='HotelReviews')
    cursor = cnx.cursor()

    # Call the stored procedure
    cursor.callproc('get_data', [limit])

    # Fetch the results
    results = []
    for result in cursor.stored_results():
        results = result.fetchall()

    # Close the cursor and connection
    cursor.close()
    cnx.close()

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results, columns=['Hotel_Name', 'Reviewer_Nationality', 'Negative_Review', 'Positive_Review'])

    return df
df = call_stored_procedure(10000)
print(df.head())

def find_duplicate_negative_reviews(df):
    # Filter the dataframe to only include duplicate negative reviews
    duplicate_negative_reviews = df[df['Negative_Review'].duplicated(keep=False)]

    # Group the filtered dataframe by 'Negative_Review' and count the occurrences
    duplicate_counts = duplicate_negative_reviews.groupby('Negative_Review').size().reset_index(name='Count')

    # Sort the resulting dataframe by 'Count' in descending order
    duplicate_counts_sorted = duplicate_counts.sort_values('Count', ascending=False)

    return duplicate_counts_sorted
duplicate_counts_sorted_negative = find_duplicate_negative_reviews(df)

def find_duplicate_positive_reviews(df):
    # Filter the dataframe to only include duplicate positive reviews
    duplicate_positive_reviews = df[df['Positive_Review'].duplicated(keep=False)]

    # Group the filtered dataframe by 'Positive_Review' and count the occurrences
    duplicate_counts = duplicate_positive_reviews.groupby('Positive_Review').size().reset_index(name='Count')

    # Sort the resulting dataframe by 'Count' in descending order
    duplicate_counts_sorted = duplicate_counts.sort_values('Count', ascending=False)

    return duplicate_counts_sorted
duplicate_counts_sorted_positive = find_duplicate_positive_reviews(df)

def update_reviews(df):
    # List of phrases to match for negative and positive reviews
    negative_phrases_to_match = [
        'Nothing', ' Nothing', 'nothing', 'None', 'N A', 'n a', 'N a', 'Nothing really',
        'Absolutely nothing', 'Nothing to dislike', 'Nothing I can think of',
        'Nothing at all', 'All good', 'No complaints'
    ]

    positive_phrases_to_match = ['No Positive', 'nothing']

    # Update the negative and positive reviews in the dataframe that match the specified phrases to None
    df['Negative_Review'] = df['Negative_Review'].apply(lambda x: None if x.strip() in negative_phrases_to_match else x)
    df['Positive_Review'] = df['Positive_Review'].apply(lambda x: None if x.strip() in positive_phrases_to_match else x)

    return df
updated_df = update_reviews(df)









# Split into two dataframes
#dfPositive = dfSQL.filter(['A','B','D'], axis=1)
#dfNegative = dfSQL.filter(['A','B','D'], axis=1)



