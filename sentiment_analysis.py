""" This example is inspired by https://hortonworks.com/tutorial/sentiment-analysis-with-apache-spark/ """
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import *
# from pyspark.mllib.feature import HashingTF
# from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, HashingTF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

global spark
tweet_path = "/jml/data/d/1 CLASS STUDY/1 Lecture/1 IT4BI Second/5 Python/Lab/sentiment_analysis/tweets"
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Sentiment analysis") \
    .getOrCreate()

# Read data into spark
tweets = spark.read.json(tweet_path)
tweets.cache()
tweets.show()

# Filter message with happy/sad word
message = tweets.select("msg")
sad_message = message.filter(message.msg.contains("sad"))
happy_message = message.filter(message.msg.contains("happy"))

# Get smallest number of sad/happy message
min_message = np.min([sad_message.count(), happy_message.count()])
union_message = sad_message.limit(min_message).unionAll(happy_message.limit(min_message))

# Add label column and remove sad|happy from message
label_message = union_message.withColumn("is_happy", when(union_message.msg.contains("happy"), 1).otherwise(0))
label_message = label_message.withColumn("msg", regexp_replace("msg", "sad|happy", ""))
# union_message.show(100)

# Hasing message
# hashingTF = HashingTF(2000)
# hashingTF = HashingTF()
# hash_message = label_message.rdd.map(lambda row: (hashingTF.transform(row[0]), row[1]))
# hash_message = spark.createDataFrame(hash_message, ["hash_msg", "is_happy"])
# hash_message = hash_message.withColumnRenamed("_1", "hash_msg") \
#     .withColumnRenamed("_2", "is_happy")
# label_message = label_message.withColumn("hash", hashingTF.transform(label_message.msg))
# label_message.show(100)
# hash_message.show()
tokenizer = Tokenizer(inputCol="msg", outputCol="token_msg")
hash_message = tokenizer.transform(label_message)
hasingTF = HashingTF(inputCol="token_msg", outputCol="hash_msg", numFeatures=2000)
hash_message = hasingTF.transform(hash_message)
# hash_message = label_message

# Split messages into training and validation set
label_indexer = StringIndexer(inputCol="is_happy", outputCol="indexed_label").fit(hash_message)
feature_indexer = VectorIndexer(inputCol="hash_msg", outputCol="indexed_hash_msg").fit(hash_message)

validation_set, training_set = hash_message.randomSplit([0.3, 0.7])
validation_set.show()
training_set.show()

# Build model using Gradient Boost
# classifier = GBTClassifier(featuresCol="indexed_hash_msg", labelCol="indexed_label", maxDepth=5, maxIter=20)
# pipeline = Pipeline(stages=[label_indexer, feature_indexer, classifier])
# model = pipeline.fit(training_set)
classifier = GBTClassifier(featuresCol="hash_msg", labelCol="is_happy", maxDepth=5, maxIter=20)
model = classifier.fit(validation_set)

# Validate test set
predictions = model.transform(validation_set)
predictions.show()

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="is_happy", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test error: %g" % (1.0 - accuracy))

# gbtModel = model.stages[2]
# print gbtModel
