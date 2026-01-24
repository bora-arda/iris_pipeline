from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import logging
import os
from core.utils import setup_logger, load_settings

config = load_settings("settings.json")
setup_logger()
logger = logging.getLogger(__name__)

def download_dataset():
    logger.info("Downloading Iris dataset")
    df = load_iris(as_frame=True)['frame']
    try:
        df.to_csv(f"{config['general']['raw_data_dir']}iris.csv")
        logger.info(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns")
    except Exception as e:
        print(e)

def split_dataset(random_state = config["general"]["random_state"], test_size = config["train"]["test_size"]):
    logger.info("Loading dataset")
    X, y = load_iris(return_X_y = True)
    logger.info("Dataset Loaded")
    logger.info("Splitting dataset")
    X_train, X_inf, y_train, y_inf = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y, shuffle = True) # Split into train and inference
    logger.info(
        f"Split into {len(X_train)} training rows "
        f"and {len(X_inf)} inference rows"
    )

    # Applying MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_inf = scaler.transform(X_inf)
    logger.info("Scaling applied")
    
    # Column stacking X and y to make them able to transform into dataframes.
    train_data = np.column_stack((X_train, y_train))
    inference_data  = np.column_stack((X_inf, y_inf))

    # Dummy feature names to seperate target for better understanding
    num_features = X.shape[1]
    feature_columns = [f"feature_{i}" for i in range(num_features)]
    columns = feature_columns + ["target"]

    # Transform into pandas dataframes before saving locally
    train_df = pd.DataFrame(train_data, columns = columns)
    inference_df  = pd.DataFrame(inference_data, columns = columns)

    return train_df, inference_df

def save_dataframes(train_df: pd.DataFrame, inference_df: pd.DataFrame, save_folder: str = config["general"]["processed_data_dir"]):
    """
    Saves created dataframes locally
    
    :param train_df: training dataframe
    :type train_df: pd.DataFrame
    :param inference_df: inference/test dataframe
    :type inference_df: pd.DataFrame
    :param save_folder: Directory of data to be saved
    :type save_folder: str
    """
    
    # Save locally as csv files
    logger.info("Saving dataframes.")
    train_data_table = os.path.join(save_folder, config['train']['data_table'])
    inference_data_table = os.path.join(save_folder, config['inference']['data_table'])
    train_df.to_csv(f"{train_data_table}", index=False)
    inference_df.to_csv(f"{inference_data_table}", index=False)
    
    logger.info(f"Dataset saved as {config['train']['data_table']} and {config['inference']['data_table']} inside {save_folder} folder.")

def main():
    try:
        download_dataset()
        train_df, inference_df = split_dataset()
        save_dataframes(train_df, inference_df)
        
        logger.info("Preparation Pipeline completed successfully.")
    except Exception as e:
        logger.exception(f"Preparation Pipeline failed due to:\n{e}")
        
        
if __name__ == "__main__":
    main()








