import matplotlib.pyplot as plt
import mysql.connector
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy as db
from sqlalchemy import text
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def csv_to_sql():
    # Load the CSV file into a pandas dataframe
    dfCsv = pd.read_csv(filepath_or_buffer='Hotel_Reviews.csv', engine='python', on_bad_lines='skip')
    # Add new columns for word counts
    dfCsv['Negative_Review_Word_Count'] = dfCsv['Negative_Review'].apply(lambda x: len(x.split()))
    dfCsv['Positive_Review_Word_Count'] = dfCsv['Positive_Review'].apply(lambda x: len(x.split()))
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
        SELECT Hotel_Name, Reviewer_Nationality, Negative_Review, Positive_Review, 
               Negative_Review_Word_Count, Positive_Review_Word_Count
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
    df = pd.DataFrame(results, columns=['Hotel_Name', 'Reviewer_Nationality', 'Negative_Review',
                                        'Positive_Review', 'Negative_Review_Word_Count', 'Positive_Review_Word_Count'])

    return df
df = call_stored_procedure(10000)

# Based the filter on this
def find_duplicate_negative_reviews(df):
    # Filter the dataframe to only include duplicate negative reviews
    duplicate_negative_reviews = df[df['Negative_Review'].duplicated(keep=False)]

    # Group the filtered dataframe by 'Negative_Review' and count the occurrences
    duplicate_counts = duplicate_negative_reviews.groupby('Negative_Review').size().reset_index(name='Count')

    # Sort the resulting dataframe by 'Count' in descending order
    duplicate_counts_sorted = duplicate_counts.sort_values('Count', ascending=False)

    return duplicate_counts_sorted
duplicate_counts_sorted_negative = find_duplicate_negative_reviews(df)

# Based the filter on this
def find_duplicate_positive_reviews(df):
    # Filter the dataframe to only include duplicate positive reviews
    duplicate_positive_reviews = df[df['Positive_Review'].duplicated(keep=False)]

    # Group the filtered dataframe by 'Positive_Review' and count the occurrences
    duplicate_counts = duplicate_positive_reviews.groupby('Positive_Review').size().reset_index(name='Count')

    # Sort the resulting dataframe by 'Count' in descending order
    duplicate_counts_sorted = duplicate_counts.sort_values('Count', ascending=False)

    return duplicate_counts_sorted
duplicate_counts_sorted_positive = find_duplicate_positive_reviews(df)

def filter_reviews(df):
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
updated_df = filter_reviews(df)


# Preprocess the data and combine reviews
def preprocess_data(df):
    # Combine negative and positive reviews into a single column
    negative_reviews = df[['Negative_Review']].copy()
    negative_reviews['Sentiment'] = 0
    negative_reviews.columns = ['Review', 'Sentiment']

    positive_reviews = df[['Positive_Review']].copy()
    positive_reviews['Sentiment'] = 1
    positive_reviews.columns = ['Review', 'Sentiment']

    combined_reviews = pd.concat([negative_reviews, positive_reviews], ignore_index=True)
    combined_reviews.dropna(subset=['Review'], inplace=True)

    return combined_reviews
combined_reviews = preprocess_data(updated_df)

def train_model(combined_reviews):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_reviews['Review'], combined_reviews['Sentiment'], test_size=0.2, random_state=42)

    # Logistic Regression
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LogisticRegression())
    ])

    # Multinomial Naive Bayes
    mnb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])

    # Support Vector Machines
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', LinearSVC())
    ])

    # Train the classifiers
    lr_pipeline.fit(X_train, y_train)
    mnb_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)

    # Evaluate the performance of the classifiers
    lr_preds = lr_pipeline.predict(X_test)
    mnb_preds = mnb_pipeline.predict(X_test)
    svm_preds = svm_pipeline.predict(X_test)

    return X_test, y_test, lr_preds, mnb_preds, svm_preds
X_test, y_test, lr_preds, mnb_preds, svm_preds = train_model(combined_reviews)

def combine_models(preds1, preds2, preds3):
    combined_preds = []
    for p1, p2, p3 in zip(preds1, preds2, preds3):
        votes = [p1, p2, p3]
        combined_preds.append(np.argmax(np.bincount(votes)))
    return combined_preds
combined_preds = combine_models(lr_preds, mnb_preds, svm_preds)
combined_accuracy = accuracy_score(y_test, combined_preds)

classifier_accuracies = {
    'Combined': combined_accuracy,
    'Logistic Regression': accuracy_score(y_test, lr_preds),
    'Multinomial Naive Bayes': accuracy_score(y_test, mnb_preds),
    'Support Vector Machines': accuracy_score(y_test, svm_preds)
}

# Calculate the accuracy of all the models and the combined model
print("Combined Accuracy: ", classifier_accuracies['Combined'])
print("Logistic Regression Accuracy: ", classifier_accuracies['Logistic Regression'])
print("Multinomial Naive Bayes Accuracy: ", classifier_accuracies['Multinomial Naive Bayes'])
print("Support Vector Machines Accuracy: ", classifier_accuracies['Support Vector Machines'])

# Plot the distribution of word counts in the positive and negative reviews
def plot_word_count_distribution(reviews):
    plt.figure(figsize=(12, 6))
    plt.hist(reviews[reviews['Sentiment'] == 0]['Review'].str.len(), alpha=0.6, bins=40, label='Negative', color='red')
    plt.hist(reviews[reviews['Sentiment'] == 1]['Review'].str.len(), alpha=0.6, bins=40, label='Positive', color='blue')
    plt.title('Word Count Distribution in Reviews')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()
plot_word_count_distribution(combined_reviews)

# Plot the frequency of hotel names
def plot_top_reviewed_hotels(df):
    plt.figure(figsize=(12, 6))
    df['Hotel_Name'].value_counts().head(20).plot(kind='barh')
    plt.title('Top Reviewed Hotels')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Hotel Name')
    plt.gca().invert_yaxis()
    plt.show()
plot_top_reviewed_hotels(df)

# Plot the frequency of reviewer nationalities
def plot_top_reviewer_nationalities(df):
    plt.figure(figsize=(12, 6))
    df['Reviewer_Nationality'].value_counts().head(20).plot(kind='barh')
    plt.title('Top 20 Reviewer Nationalities')
    plt.xlabel('Number of Reviews')
    plt.ylabel('Nationality')
    plt.gca().invert_yaxis()
    plt.show()
plot_top_reviewer_nationalities(df)

# Plot the accuracy of the classifiers
def plot_classifier_accuracies(accuracies):
    plt.figure(figsize=(8, 6))
    plt.ylim(0.7, 1.0)
    plt.bar(range(len(accuracies)), list(accuracies.values()), align='center', color='purple')
    plt.xticks(range(len(accuracies)), list(accuracies.keys()), rotation=45)
    plt.title('Accuracy of Classifiers')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.show()
plot_classifier_accuracies(classifier_accuracies)
