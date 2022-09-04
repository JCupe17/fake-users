import argparse
import pickle
import logging

import numpy as np
import sklearn.pipeline

import config
import pandas as pd
import sys

import src.utils as utils

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)


def model_selection(X_train: pd.DataFrame, y_train: pd.DataFrame) -> sklearn.model_selection.GridSearchCV:
    """Performs the training and selection of a model based on f1 score."""
    # Defining the pipeline
    # This pipeline input first missing data, then it scales the data to avoid any problem with low and high values,
    # and finally it trains a model
    pipe = Pipeline([('inputer', SimpleImputer(strategy="constant", fill_value=0)),
                     ('scaler', StandardScaler()),
                     ('classifier', LogisticRegression())])

    # Defining a search space for a GridSearch
    search_space = [{'classifier': [LogisticRegression(solver='lbfgs')],
                     'classifier__C': [0.01, 0.1, 1.0]},
                    {'classifier': [RandomForestClassifier(n_estimators=100)],
                     'classifier__max_depth': [5, 10, None]}]

    # The GridSearch is based on the F1 score since we are facing an unbalanced dataset
    clf = GridSearchCV(pipe, search_space, cv=10, verbose=0, scoring='f1')
    clf = clf.fit(X_train, y_train)

    return clf


def compute_metrics(y: list, y_pred_proba: list, threshold: float) -> [float, np.ndarray]:
    if np.shape(y_pred_proba)[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
    y_pred = (y_pred_proba > threshold).astype('float')
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    return f1, cm


def compute_recommended_threshold(y: list, y_pred_proba: list) -> float:
    """Computes the better threshold that optimizes the F1 score based on the precision recall score."""
    # Computes the precision recall and thresholds using the PR curve
    if np.shape(y_pred_proba)[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
    precision, recall, thresholds_prc = precision_recall_curve(y, y_pred_proba)
    # Computes the F1 score for each tuple precision and recall
    f1_scores = (2 * precision * recall) / (precision + recall)
    # Gets the index of the largest F1 score
    ix = np.argmax(f1_scores)
    return thresholds_prc[ix]


def main(input_file: str, output_file: str = None):
    # 1. Load the input file
    logging.info(f"Loading the input file {input_file} ...")
    df = utils.load_input_file(filename=input_file)
    logging.info(f"Input dataframe contains {len(df)} lines")
    # 2. Check the input columns
    logging.info(f"Checking the columns of the input dataframe for the training ...")
    if not utils.check_input_columns(df, config.TRAIN_INPUT_COLUMNS):
        sys.exit("Check the input columns of the training input file")
    # 3. Feature Engineering
    logging.info(f"Feature engineering ...")
    df = utils.feature_engineering(df, training=True)
    # 4. Data Split
    logging.info(f"Splitting data in train and test set ...")
    X = df[config.FEATURES]
    y = np.ravel(df[[config.TARGET]].values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    logging.info(f"The training set has {len(X_train)} samples and the test set has {len(X_test)} samples")
    # 5. Model Selection
    logging.info(f"Starting training and model selection ...")
    clf = model_selection(X_train, y_train)
    logging.info(f"Best estimator: {clf.best_estimator_}")
    logging.info(f"Best score on the training set: {clf.best_score_}")
    # 6. Compute metrics for a default threshold (0.5 for a binary classification)
    best_model = clf.best_estimator_
    y_test_pred_proba = best_model.predict_proba(X_test)
    f1, cm = compute_metrics(y_test, y_test_pred_proba, threshold=0.5)
    logging.info(f"Test set: F1 score with a default threshold = {f1}")
    logging.info(f"Test set: Confusion matrix: \n {cm}")
    # 7. Compute recommended threshold
    recommended_threshold = compute_recommended_threshold(y_test, y_test_pred_proba)
    recommended_f1, recommended_cm = compute_metrics(y_test, y_test_pred_proba, threshold=recommended_threshold)
    logging.info(f"Test set: Recommended threshold for the binary classification = {recommended_threshold}")
    logging.info(f"Test set: Recommended F1 score with the best threshold = {recommended_f1}")
    logging.info(f"Test set: Confusion matrix based on the recommended threshold: \n {recommended_cm}")
    # 7. Save the pipeline and the recommended threshold
    if output_file is None:
        output_file = config.MODEL_FILEPATH
    logging.info(f"Saving the model and the recommended threshold in the file {output_file} ...")
    with open(output_file, "wb") as f:
        pickle.dump([best_model, recommended_threshold], f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for training a model.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path of the input file to train a model."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path of the output file to save the model and the recommended threshold.",
    )
    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)
