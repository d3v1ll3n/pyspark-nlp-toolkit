from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, lower
from pyspark.sql.types import ArrayType, StringType
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Spark session
spark = SparkSession.builder.appName("NLPPreprocessing").getOrCreate()

# Options for only showing the Errors or Warnings
spark.sparkContext.setLogLevel("ERROR")
# spark.sparkContext.setLogLevel("WARN")


# Locating and loading the dataset path
dataset_path = "/home/ivarr/Downloads/title.basics.tsv"
dataset = spark.read.option("header", "true").option("delimiter", "\t").csv(dataset_path)

# Selecting the 'titleType' and 'primaryTitle' columns from the dataset
selected_data = dataset.select("titleType", "primaryTitle")

# Filtering the dataset to keep only movie titles
movies_data = selected_data.filter(selected_data.titleType == "movie")

# Normalization
normalized_data = movies_data.withColumn("normalizedText", lower(movies_data.primaryTitle))

# Tokenization
def nltk_tokenize(text):
    return word_tokenize(text)

tokenize_udf = udf(nltk_tokenize, ArrayType(StringType()))
tokenized_data = normalized_data.withColumn("tokens", tokenize_udf(normalized_data.normalizedText))

# Removing stop words
stopwords = stopwords.words("english")
stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filteredTokens", stopWords=stopwords)
filtered_data = stopwords_remover.transform(tokenized_data)

# Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

lemmatize_udf = udf(lemmatize_tokens, ArrayType(StringType()))
lemmatized_data = filtered_data.withColumn("lemmatizedTokens", lemmatize_udf(filtered_data.filteredTokens))

# Term frequency (TF) computation
cv = CountVectorizer(inputCol="lemmatizedTokens", outputCol="rawFeatures")
cv_model = cv.fit(lemmatized_data)
featurized_data = cv_model.transform(lemmatized_data)

# Inverse document frequency (IDF) computation
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
preprocessed_data = idf_model.transform(featurized_data)

# Word embedding using Word2Vec
word2vec = Word2Vec(vectorSize=100, inputCol="lemmatizedTokens", outputCol="wordVectors")
word2vec_model = word2vec.fit(lemmatized_data)
preprocessed_data = word2vec_model.transform(preprocessed_data)

# Showing the preprocessed data
preprocessed_data.show(truncate=False)

# Stopping the Spark session
spark.stop()
