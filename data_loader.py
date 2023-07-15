from pyspark.sql import SparkSession

def load_dataset(dataset_path):
    spark = SparkSession.builder.appName("NLPPreprocessing").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    dataset = spark.read.option("header", "true").option("delimiter", "\t").csv(dataset_path)
    return dataset
