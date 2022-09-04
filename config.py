TRAIN_INPUT_COLUMNS = ["UserId", "Event", "Category", "Fake"]
INFERENCE_INPUT_COLUMNS = ["UserId", "Event", "Category"]

FEATURES = ["count_event", "count_category"]
TARGET = "Fake"

MODEL_FILEPATH = "model/model.pkl"
