import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Function that loads the data from both csv files and merges it '''
    # Read input data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge input data
    df = pd.concat([messages, categories], axis=1)  # outer join is by default done
    return df


def clean_data(df):
    ''' Expand categorie names and remove duplicates from the dataframe '''
    # Expand categories column
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = list(
        map(lambda x: x[:-2], row))  # apply lambda function to retrive the string up to the second last character
    categories.columns = category_colnames  # Renaming of the columns
    # For the rows we are only interested in the last character, as an int
    for column in categories:
        string_value = categories[column].astype('str')
        string_value = string_value.replace(to_replace=r'.+-(\d)', value=r'\1',
                                            regex=True)  # regex for catching a single digit after a -
        # convert column from string to numeric
        categories[column] = string_value.astype('int')
    # Now we drop the original column and replace it with the expanded ones
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    # Pandas drop_duplicates method remove duplicate rows, so we need to transpose the dataframe for removing the duplicate columns and transpose back
    df_clean = df.T.drop_duplicates().T
    # Some entries have a 2 in the related column, which is not expected and assumed to be a typo
    df_clean = df_clean[df_clean.related != 2]
    return df_clean


def save_data(df, database_filename):
    ''' Save to SQLite database '''
    engine = create_engine('sqlite:///%s' % database_filename)
    df.to_sql('Messages', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()