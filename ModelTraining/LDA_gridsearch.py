import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LsiModel, Phrases, LdaModel, TfidfModel, LdaMulticore
from gensim.utils import simple_preprocess
from gensim.matutils import corpus2csc
from gensim.models import CoherenceModel
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from gensim.test.utils import datapath
import pickle
from scipy.stats import randint
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Data
train_data = pd.read_csv("C:\\Users\\RaahimSiddiqi\\Desktop\\Code\\VSC\\FYP\\DataCollection\\Data\\train-clean-customstopwords.csv")
training_data = train_data['document'].apply(lambda x: simple_preprocess(x))

# Load LDA Model
folder_name = "lda-combinedv2-80topics-standardfilter-10pass-20iters-0.56cv"
model_path = datapath(f"C:\\Users\\RaahimSiddiqi\\Desktop\\Code\\VSC\\FYP\\Models\\{folder_name}\\model")
ldamodel=LdaModel.load(model_path)

# Load LDA Dictionary
dictionary_path = datapath(f"C:\\Users\\RaahimSiddiqi\\Desktop\\Code\\VSC\\FYP\\Models\\{folder_name}\\model.id2word")
dictionary = Dictionary.load(dictionary_path)

# LDA Inference
corpus = [dictionary.doc2bow(doc) for doc in training_data]
train_probs = ldamodel[corpus]
train_probs = corpus2csc(train_probs).T.toarray()

topic_columns = [f"topic{i+1}" for i in range(train_probs.shape[1])]
train_data[topic_columns] = pd.DataFrame(train_probs, columns=topic_columns)

# Separate features and class
X = train_data[topic_columns]
y = train_data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a parameter distribution for GridCV
param_dist = {
    'n_estimators': [200, 250, 300, 350],  # Number of boosting rounds
    'learning_rate': [0.05, 0.1, 0.2],  # Step size shrinkage to prevent overfitting
    'max_depth': [7, 8, 9],  # Maximum depth of individual trees
    'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight (hessian) needed in a child
    'gamma': [0.6, 0.7, 0.8, 0.9],  # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': [0.7, 0.8, 0.9, 1.0],  # Fraction of samples used for fitting the trees
    'colsample_bytree': [0.3, 0.4, 0.5],  # Fraction of features used for building trees
    'lambda': [0.1, 1, 2],  # L2 regularization term on weights (ridge regularization)
    'alpha': [0, 0.1, 0.2, 0.5],  # L1 regularization term on weights (lasso regularization)
}

# Initialize the XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42)

# best_params = {'colsample_bytree': 0.5, 'gamma': 1, 'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 179, 'subsample': 0.9}
best_params = {}

if len(best_params) == 0:
    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_dist, cv=5, scoring='accuracy', verbose=True, n_jobs=16)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

# Train the XGBoost classifier with the best hyperparameters
best_xgb_classifier = XGBClassifier(random_state=42, **best_params)
best_xgb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_xgb_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Best Hyperparameters (Grid Search):")
print(best_params)

print("\nModel Evaluation:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

pickle.dump(best_xgb_classifier, open("lda-XGBoost-gridsearch", 'wb'))