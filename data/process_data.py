"""
PRE-PROCESSING DATA
Disaster Response Pipeline Project
Udacity - Data Science Nanodegree

How to run this script
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

INPUT:
    1) csv file containing messages (disaster_messages.csv)
    2) csv file containing categories (disaster_categories.csv)
    3) sql destination database (DisasterResponse.db)
    
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT:
        messages_filepath: messages dataset file path.
        categories_filepath: categories dataset file path.
    OUTPUT:
        df: The merged dataset
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    '''
    INPUT:
        df: The merged dataframee (raw data).
    OUTPUT:
        df: The clean dataframe (clean data).
    '''
    
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(subset = 'id', inplace = True)
    
    return df


def save_data(df, database_filepath):
    '''
    INPUT:
        df: clean data
        database_filepatch: db destination path 
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('tbl_Disaster', engine, index=False,if_exists='replace')



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
