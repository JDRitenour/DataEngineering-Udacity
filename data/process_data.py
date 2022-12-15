# import libraries
import pandas as pd
from sqlalchemy import create_engine


def load_data():
    """
    Loads the data from messages.csv and categories.csv files and merges it in one data frame

    Returns:
        df
    """
    messages = pd.read_csv('messages.csv')
    categories = pd.read_csv('categories.csv')
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Cleans data for loading into databases.

    :param df:
    :return: df
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # replace any value of '2' with a value of '1'
        categories[column] = categories[column].str.replace('2', '1')
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')

    # drop the original categories column from `df`
    df.drop(columns=['categories', 'id'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df):
    """
    Loads data to a SQLite Database

    :param df:
    """
    # load to sqlite database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    df = load_data()
    df = clean_data(df)
    save_data(df)


main()