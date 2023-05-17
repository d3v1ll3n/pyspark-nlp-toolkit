# NLP Preprocessing using PySpark

This code demonstrates an NLP preprocessing pipeline using PySpark. The pipeline includes text normalization, tokenization, stop word removal, lemmatization, term frequency (TF) computation, inverse document frequency (IDF) computation, and word embedding using Word2Vec.

## Requirements

- PySpark
- NLTK (Natural Language Toolkit)

Make sure you have the necessary libraries installed before running the code.

## Usage

Import the required libraries:

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, lower
from pyspark.sql.types import ArrayType, StringType
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
```
Create a Spark session:

```python
spark = SparkSession.builder.appName("NLPPreprocessing").getOrCreate()
```

Set the log level for Spark (OPTIONAL):

```python
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setLogLevel("WARN")
```
Load the dataset:

```python
dataset_path = "/path/to/dataset.csv"
dataset = spark.read.option("header", "true").csv(dataset_path)
```

Select the relevant columns from the dataset:

```python
selected_data = dataset.select("titleType", "primaryTitle")
```

Filter the dataset to keep only movie titles:

```python
movies_data = selected_data.filter(selected_data.titleType == "movie")
```

Normalize the text:

```python
normalized_data = movies_data.withColumn("normalizedText", lower(movies_data.primaryTitle))
```

Tokenize the text using NLTK:

```python
def nltk_tokenize(text):
    return word_tokenize(text)

tokenize_udf = udf(nltk_tokenize, ArrayType(StringType()))
tokenized_data = normalized_data.withColumn("tokens", tokenize_udf(normalized_data.normalizedText))
```

Remove stop words:

```python
stopwords = stopwords.words("english")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filteredTokens", stopWords=stopwords)
filtered_data = stopwords_remover.transform(tokenized_data)
```

Lemmatize the tokens:

```python
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

lemmatize_udf = udf(lemmatize_tokens, ArrayType(StringType()))
lemmatized_data = filtered_data.withColumn("lemmatizedTokens", lemmatize_udf(filtered_data.filteredTokens))
```

Compute term frequency (TF):

```python
cv = CountVectorizer(inputCol="lemmatizedTokens", outputCol="rawFeatures")
cv_model = cv.fit(lemmatized_data)
featurized_data = cv_model.transform(lemmatized_data)
```

Compute inverse document frequency (IDF):

```python
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
preprocessed_data = idf_model.transform(featurized_data)
```

Perform word embedding using Word2Vec:

```python
word2vec = Word2Vec(vectorSize=100, inputCol="lemmatizedTokens", outputCol="wordVectors")
word2vec_model = word2vec.fit(lemmatized_data)
preprocessed_data = word2vec_model.transform(preprocessed_data)
```

Show the preprocessed data:

```python
preprocessed_data.show(truncate=False)
```

Stop the Spark session:

```python
spark.stop()
```

Feel free to modify and adapt the code according to your specific requirements.

## License

This project is licensed under [CC BY-NC-SA license](license). Please refer to the LICENSE file for more details.


This project utilizes the following libraries:

- [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) - Apache Spark's Python API
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit for Python
- [WordNet](https://wordnet.princeton.edu/) - Lexical database for the English language

For any questions or further information, please contact [Ivarr Vinter](mailto:ivarrvinter@gmail.com).
