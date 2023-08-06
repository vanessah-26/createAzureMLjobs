# Import libraries

import argparse
import glob
import os
# added here
import mlflow    
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
#from mlflow.utils.environment import _mlflow_conda_env
from sklearn.model_selection import train_test_split



import pandas as pd

from sklearn.linear_model import LogisticRegression


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog(log_models= True)   #enable the model to aumatically log, if False, will prevent from automatically log, as it is done manually later
    model = XGBClassifier(use_label_encoder = False, eval_metrics = "logloss")
    model.fit(X_train, y_train, eval_set = [(X_test, y_test)], verbose = False)  #when the fit() method is called on the pipeline object, the model will be autologged
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    


    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(X_train, y_train, X_test, y_test):
    X_train, y_train, X_test, y_test = train_test_split(X,y, test_size = 0.4, random_state = 1) 

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
