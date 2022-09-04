import argparse
import sys
import pickle
import logging
import numpy as np

import config
import src.utils as utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)


def inference(input_file, output_file, model_path):

    # 1. Load the input file
    logging.info(f"Loading the input file {input_file} ...")
    df = utils.load_input_file(filename=input_file)
    # 2. Check the columns of the input file
    logging.info(f"Checking if the required columns for the inference are present ...")
    if not utils.check_input_columns(df, config.INFERENCE_INPUT_COLUMNS):
        sys.exit(
            f"Check the columns of your input file, it should contains the columns {config.INFERENCE_INPUT_COLUMNS}"
        )
    # 3. Load the model
    logging.info(f"Loading the model from {model_path} file ...")
    if model_path is None:
        model_path = config.MODEL_FILEPATH
    with open(model_path, "rb") as f:
        model, recommended_threshold = pickle.load(f)
    # 3. Perform the feature engineering
    logging.info(f"Performing the feature engineering ...")
    df = utils.feature_engineering(df, training=False)
    # 4. Perform the inference
    logging.info(f"Computing the prediction probability ...")
    predicted_probability = model.predict_proba(df[config.FEATURES])
    print(np.shape(predicted_probability))
    df["is_fake_probability"] = predicted_probability[:, 1]
    # 5. Save inference
    logging.info(f"Saving the results in the file {output_file} ...")
    df = df[["UserId", "is_fake_probability"]]
    df.to_csv(output_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path of the input file used to make some prediction."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path of the output file that contains the prediction probability."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path of the model file that contains the model and the recommended threshold.",
    )

    args = parser.parse_args()

    inference(input_file=args.input_file, output_file=args.output_file, model_path=args.model_path)
