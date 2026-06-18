import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

import logging

# logging configure
# now the flag would be set in the dvc
# -n  name
# -d src/data_ingestion.pz
# -p params skip
# -o output

#to add the stage : dvc stage add -n data_ingestion -d sec/data_ingestion.py -o data/raw python src/data_ingestion.py 

#logger object - Main object which is controling all of the logging action
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

#handler object - Where to write the infomration when error are showing
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

#Formatter object - How message will be published
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Conneting formatter to the handler object
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#Adding handler to the logger 
logger.addHandler(console_handler)
logger.addHandler(file_handler) 

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['make_dataset']['test_size']
        logger.debug('test size retrievesd')
        return test_size
    except FileNotFoundError:
        logger.error('File not found')
        raise
    except yaml.YAMLError as e:
        logger.error('yaml error')
        raise
    except Exception as e:
        logger.error('some error occured')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('data loaded')
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the CSV file from {data_url}.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading the data.")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])
        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logger.debug(f"only two class has been considered for the future work")
        return final_df
    except KeyError as e:
        logger.error(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        logger.error(f"Error: An unexpected error occurred during preprocessing.")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.debug(f"train and test data has been saved")
    except Exception as e:
        logger.error(f"An unexpected error occurred while saving the data.")
        raise

def main():
    try:
        test_size = load_params(params_path='params.yaml')
        df = load_data(data_url='https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path=os.path.join(os.getcwd(), 'DVC_project/data'))
    except Exception as e:
        logger.error("Failed to complete the data ingestion process.")

if __name__ == '__main__':
    main()
