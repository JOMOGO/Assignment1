import pandas as pd
import sqlalchemy as db

# create db first in MySQL
engine = db.create_engine('mysql+mysqlconnector://root:Rg123456@localhost/HotelReviews')
conn = engine.connect()

# Read out csv file with pandas
df = pd.read_csv('Hotel_Reviews.csv', sep=',', engine='python', on_bad_lines='skip')

# Create table object
df.to_sql(name='reviews', con=engine, if_exists='replace', chunksize=1000)
df.head()
conn.close()
