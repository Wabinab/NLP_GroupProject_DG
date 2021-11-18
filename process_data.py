import os
import sys
import numpy as np
import pandas as pd
# from sqlalchemy import create_engine

def load_data(filepath):

    '''
    
    
    
    '''
    BASE_DIR = filepath
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroups')

    # This code from https://www.kaggle.com/mansijharia
    # May take a lot of time 

    texts = []
    labels_index = {}
    labels = []

    for name in sorted(os.listdir((BASE_DIR+'20_newsgroups'))):
        path = os.path.join(BASE_DIR,'20_newsgroups', name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                    with open(fpath, **args) as f:
                        t = f.read()
                        #this is to skip the metadata on the first paragraph, may and may not use it
                        #i = t.find('\n\n')
                        #if 0 < i:
                        #   t = t[i:]
                        texts.append(t)
                    labels.append(label_id)

    return texts, labels, labels_index                    


def clean_data(df):
    # in progress
    # drop duplicates
    df.drop_duplicates()
    return df
    


def save_data(df, database_filename):
    None
    #creat a database
    # engine = create_engine('sqlite:///database_filename)
    #save into adatbase
    # table_name = database_filename.replace(".db","") + "_table"
    # df.to_sql(table_name, engine, index=False, if_exists='replace')
 


def main():
    if len(sys.argv) ==2:

        filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n: {}'.format(filepath))
        texts, labels, labels_index = load_data(filepath)

        # print('Cleaning data...')
        # df = clean_data(df)
        
        # print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        # save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the folder paths of 20_newsgroups'\
              'datasets as the first and the second argument '\
              'the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              '20_newsgroups data_cleaning.db')


if __name__ == '__main__':
    main()