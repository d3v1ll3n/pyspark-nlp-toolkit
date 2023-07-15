from pyspark.sql.functions import udf, lower
from pyspark.sql.types import ArrayType, StringType
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import Word2Vec
from pyspark.ml import Pipeline

def normalize_data(dataframe):
    normalized_data = dataframe.withColumn("normalizedText", lower(dataframe.primaryTitle))
    return normalized_data

def nltk_tokenize(text):
    return word_tokenize(text)

def tokenize_data(dataframe):
    nltk_tokenize_udf = udf(nltk_tokenize, ArrayType(StringType()))
    tokenized_data = dataframe.withColumn("tokens", nltk_tokenize_udf(dataframe.normalizedText))
    return tokenized_data

def remove_stopwords(dataframe):
    stopwords_list = stopwords.words("english")
    stopwords_remover = StopWordsRemover(inputCol="tokens", outputCol="filteredTokens", stopWords=stopwords_list)
    filtered_data = stopwords_remover.transform(dataframe)
    return filtered_data

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def lemmatize_data(dataframe):
    lemmatize_udf = udf(lemmatize_tokens, ArrayType(StringType()))
    lemmatized_data = dataframe.withColumn("lemmatizedTokens", lemmatize_udf(dataframe.filteredTokens))
    return lemmatized_data

def compute_tf(dataframe):
    cv = CountVectorizer(inputCol="lemmatizedTokens", outputCol="rawFeatures")
    cv_model = cv.fit(dataframe)
    featurized_data = cv_model.transform(dataframe)
    return featurized_data

def compute_idf(dataframe):
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(dataframe)
    preprocessed_data = idf_model.transform(dataframe)
    return preprocessed_data

def compute_word2vec(dataframe):
    word2vec = Word2Vec(vectorSize=100, inputCol="lemmatizedTokens", outputCol="wordVectors")
    word2vec_model = word2vec.fit(dataframe)
    preprocessed_data = word2vec_model.transform(dataframe)
    return preprocessed_data

def preprocess_data(dataframe):
    normalized_data = normalize_data(dataframe)
    tokenized_data = tokenize_data(normalized_data)
    filtered_data = remove_stopwords(tokenized_data)
    lemmatized_data = lemmatize_data(filtered_data)
    featurized_data = compute_tf(lemmatized_data)
    preprocessed_data = compute_idf(featurized_data)
    preprocessed_data = compute_word2vec(preprocessed_data)
    return preprocessed_data
