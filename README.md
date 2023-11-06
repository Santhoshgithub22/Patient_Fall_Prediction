# PATIENT FOOTFALL PREDICTION

![Patient Fall](https://media.istockphoto.com/id/1340235916/photo/female-asian-patient-fell-lying-on-the-floor-at-hospital-she-is-trying-to-raise-their-hand.jpg?s=612x612&w=0&k=20&c=8LqmResuX-iHN-wBBIfRw0AazDccxtLMiA7SVLw6g14=)

## Table of Contents
- [Introduction](#introduction)
- [Prerequisite](#prerequisite)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Importance](#feature-importance)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

### Introduction

Patient Fall Prediction is a critical healthcare application that uses data science and machine learning to anticipate and prevent patient falls. By analyzing patient data, environmental factors, and sensor inputs, this technology provides early warnings to healthcare providers, ensuring patient safety, and better care. Its primary goal is to create a safer patient environment, improve healthcare quality, and optimize resource allocation within medical facilities. This innovative application enhances patient care and represents a significant advancement in healthcare technology.

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.7 or higher
- Pip (Python package manager)
- Virtual environment (recommended)

### Dataset

Explain how to access and preprocess the data used for training and prediction. [kaggle dataset link]("https://www.kaggle.com/code/saadmansakib/human-fall-detection-using-random-forest-97-47")

### Installation

To install the required dependencies, use the following command:

```shell
pip install -r requirements.txt
```

### Methodology

**Data Collection:**

Gather a diverse dataset that includes sensor data, such as accelerometer and gyroscope readings, from wearable devices.
Include data from both fall and non-fall scenarios to ensure a balanced dataset.

**Data Preprocessing:**

Clean the data by handling missing values and outliers.
Normalize or standardize the data to ensure all features have the same scale.

**Feature engineering:** Extract relevant features from raw sensor data, such as mean, variance, skewness, and kurtosis.

**Data Splitting:**

Split the dataset into training and testing sets. A common split is 70-30 or 80-20 for training and testing, respectively.
Model Selection:

Used machine learning models for fall detection, such as Logistic Regression, Decision Trees, Random Forest Classifier, Support Vector Classifier, Naive Bayes, Gradient Boosting, Ada Boosting, & K-Nearst Neighbours.

**Model Training:**

Train the selected models on the training dataset using the preprocessed data.

**Model Evaluation:**

**Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and area under the receiver operating characteristic (ROC-AUC) curve.
Use confusion matrices to analyze false positives and false negatives.
Model Optimization:

Deployment:

Deployed by using FLASK and deployed in the cloud by using AWS Beanstalk for healthcare system processing for fall detection.

Update the model as new data becomes available to improve accuracy.

