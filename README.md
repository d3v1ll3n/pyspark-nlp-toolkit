# PySpark NLP Preprocessing

This repository contains PySpark code for performing natural language processing (NLP) preprocessing tasks. The code leverages PySpark's distributed processing capabilities to handle large-scale datasets and applies various NLP techniques such as normalization, tokenization, stop word removal, lemmatization, TF-IDF computation, and word embedding using Word2Vec.

## Code Structure

The code is organized into separate files based on their functionality:

- `data_loader.py`: Contains functions for loading the dataset using SparkSession and configuring the dataset path.
- `preprocessing.py`: Implements functions for NLP preprocessing tasks, including normalization, tokenization, stop word removal, lemmatization, and feature extraction using TF-IDF and Word2Vec.
- `main.py`: Orchestrates the data loading, preprocessing, and displays the preprocessed data.

## Usage

To use this code, follow the steps below:

1. Install the required dependencies:
   - PySpark: Ensure that you have Apache Spark installed and properly configured.
   - NLTK: Install the NLTK library and download the required resources.

2. Set the dataset path:
   - Update the `dataset_path` variable in `main.py` with the path to your dataset file.

3. Run the code:
   - Execute `main.py` to perform the NLP preprocessing tasks on the dataset.
   - The preprocessed data will be displayed in the console.

## Requirements

- Apache Spark (https://spark.apache.org)
- NLTK (https://www.nltk.org)

## Dataset

The code expects a TSV-formatted dataset file containing information about titles. By default, the code filters the dataset to keep only movie titles based on the 'titleType' column.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
