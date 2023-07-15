from data_loader import load_dataset
from preprocessing import preprocess_data

dataset_path = '/path/to/dir'
dataset = load_dataset(dataset_path)

preprocessed_data = preprocess_data(dataset)

preprocessed_data.show(truncate=False)

spark.stop()
