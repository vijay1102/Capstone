# Capstone Project: Sparkify

# Udacity Datascience Nano Degree Capstone Project:

Pyspark code that explores and cleans a large dataset, engineers features and labels, and trains a model to predict which "Sparkify" music service users are likely to cancel their subscriptions.

The code was developed with a Jupyter Notebook on the Apache Spark framework running on an AWS cluster.

# Motivation:

The overall purpose of the project is to use the Apache Spark framework on an AWS cluster for training a machine learning model with a large dataset. The goal I hoped to accomplish while learing Spark, was to develop an accurate model for predicting when a "Sparkify" user is likely to churn.

# Blog Post:

The full discussion of my project and code are in the attached notebook. There I've described all the details about the project and why those steps has taken. Below are some key details about the project.

1. In this notebook, I've implemented a model trying to predict customers who will be churned in near future based on various factors available in given dataset. 

2. As part of data cleaning, I've removed rows where no userId was present, converted timestamp to a human readable format, converted gender to binary numeric column. 

3. In total 10 features were built for modeling purpose but given the fact that my model was taking much time to process, I've selected some key features to process in model. 

4. I've used 4 modeling techniques: logistic regression, GBM & SVM and selected GBM as the final model implemented for predicting final result. 

5. I've used cross validation and grid search methods to fine tune my model. here, I've achieved about 70% of accuracy, and F1 score of 0.65 , which is about 3% improvement compare to my baseline model for sending everyone an offer.

**Reflection:**

- This project gives me an exposure to spark environment that how we can analyze a large volume of data on a personal laptop which may not be capable enough to analyze. 

- It happened in my case where this smaller data also has take large amount of processing time, hence I've skipped my thought to rerun the process on the full data.

- By identifying customer with high chance of getting churned prior to the acutal losing, companies can use this information & wil be equipped better to retain them at minimal cost by using targeted messages and offers only to that pool of customers.

- One of the interesting yet difficult thing during the project was brainstroming the features that will be useful for actual modeling exercise. Developing useful features is crucial for developing a good model, and requires a lot of energy and efforts. Explanatory and exploratory data analysis play important role in this process.

**Improvement:**

- These features surely has enough room for improvement after considering more factors into analysis, adding more domain knowledges and expertise. Considering the fact that our model's data inputs for training, testing & validating has very less data (139, 79, 7) respectively, it has a huge potential to improve if the sample size increases, and the expected performance will also increase.

# What's Inside:

Sparkify.html : Jupyter Notebook with code running on my local machine with a small subset of the total dataset. This code was used to explore and clean the data and tryout engineering the features before moving to the AWS cluster.

Sparkify.ipynb: Jupyter Notebook with code for running on the AWS cluster. This is the same Jupyter notebook which is available in .html format and does not contains the complete dataset. This notebook has used sample dataset for analysis and modeling.

Credits

I relied on three main references during this project:

1. The Apache Spark documentation and examples for linear regression
2. Online discussion fourms like stackoverflow.com

# Libraries Used:

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, concat, desc, explode, lit, min, max, split, udf, isnull
from pyspark.sql import functions as sF
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, RegexTokenizer, StandardScaler, StopWordsRemover, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Window
from pyspark.sql.functions import sum as Fsum
