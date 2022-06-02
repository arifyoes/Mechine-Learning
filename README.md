# Mechine-Learning
# Library

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from IPython.display import display

# Feature Engineering
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce

# Model Selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve
from sklearn.model_selection import RandomizedSearchCV

# Resampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

# Imbalance Dataset
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

# Ignore Warning
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Set max columns
pd.set_option('display.max_columns', None)

#Load Dataset
df = pd.read_csv("data_hotel_booking_demand.csv")

df.head()

df.tail()

# Data Cleaning

# Count and Datatype for each Column
df.info()

listItem = []
for col in df.columns :
    listItem.append([col, df[col].dtype, df[col].isna().sum(), round((df[col].isna().sum()/len(df[col])) * 100,2),
                    df[col].nunique(), list(df[col].drop_duplicates().sample(2).values)]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
dfDesc

*Check Every Columns*

df['country'].unique()

df.dropna()['market_segment'].unique()

df.dropna()['deposit_type'].unique()

df.dropna()['days_in_waiting_list'].unique()

df.dropna()['customer_type'].unique()

df.dropna()['reserved_room_type'].unique()

df.dropna()['booking_changes'].unique()

df.dropna()['is_canceled'].unique()

*Missing Values*

df.isna().sum()/len(df.index)*100

# PreProcessing

*Preprocessing Scheme*

- OneHotEncoding: country, market_segment, deposit_type, customer_type, reserved_room_type 
- Manual: previous_cancellations, booking_changes, days_in_waiting_list, required_car_parking_spaces, total_of_special_requests.

mode_onehot_pipe = Pipeline([
    ('encoder', SimpleImputer(strategy = 'most_frequent')),
    ('one hot encoder', OneHotEncoder(handle_unknown = 'ignore'))
])

transformer_scale = ColumnTransformer([
    ('one hot', OneHotEncoder(handle_unknown = 'ignore'), ['country', 'market_segment', 'deposit_type', 'customer_type', 'reserved_room_type']),], remainder = 'passthrough')

df['is_canceled'].value_counts()

Dari dataset, ada indikasi imbalance data

df['is_canceled'].value_counts()/df.shape[0]*100

* *0 = cancel*
* *1 = tidak cancel*

        - TN: Ada kosumen yang di prediksi cancel dan kenyataannya cancel
        - TP: Ada kosumen yang di prediksi tidak cancel dan kenyataannya tidak cancel
        - FP: Ada kosumen yang di prediksi cancel dan kenyataannya tidak cancel
        - FN: Ada kosumen yang di prediksi tidak cancel dan kenyataanya cancel

Tindakan:
* FP: konsumen 
* FN: 

- > Yang akan di tekan adalah FP, namun akan mencoba roc_auc_score

*Splitting Data*

X = df.drop('is_canceled', axis = 1)
y = df['is_canceled']

X.shape

X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,
                                                   stratify = y,
                                                    test_size = 0.3,
                                                   random_state = 2727)

# Modeling

*Define Model*

logreg = LogisticRegression()
tree = DecisionTreeClassifier(random_state = 2727)
knn = KNeighborsClassifier()
rf = RandomForestClassifier(random_state = 2727)



# Cross Validation

logreg_pipe_scale = Pipeline([
    ('transformer', transformer_scale),
    ('logreg', logreg)
])

tree_pipe_scale = Pipeline([
    ('transformer', transformer_scale),
    ('tree', tree)
])

knn_pipe_scale = Pipeline([
    ('transformer', transformer_scale),
    ('knn', knn)
])

rf_pipe_scale = Pipeline([
    ('transformer', transformer_scale),
    ('rf', rf)
])

def model_evaluation(model, metric):
    skfold = StratifiedKFold(n_splits = 5)
    model_cv = cross_val_score(model, X_train_val, y_train_val, cv = skfold, scoring = metric)
    return model_cv

logreg_pipe_scale_cv = model_evaluation(logreg_pipe_scale, 'roc_auc')
tree_pipe_scale_cv = model_evaluation(tree_pipe_scale, 'roc_auc')
knn_pipe_scale_cv = model_evaluation(knn_pipe_scale, 'roc_auc')
rf_pipe_scale_cv = model_evaluation(rf_pipe_scale, 'roc_auc')

for model in [logreg_pipe_scale, tree_pipe_scale, knn_pipe_scale, rf_pipe_scale]:
    model.fit(X_train_val, y_train_val)

score_mean = [logreg_pipe_scale_cv.mean(), tree_pipe_scale_cv.mean(), knn_pipe_scale_cv.mean(),
              rf_pipe_scale_cv.mean()]
score_std = [logreg_pipe_scale_cv.std(), tree_pipe_scale_cv.std(), knn_pipe_scale_cv.std(),
             rf_pipe_scale_cv.std()]
score_roc_auc = [roc_auc_score(y_test, logreg_pipe_scale.predict(X_test)),
            roc_auc_score(y_test, tree_pipe_scale.predict(X_test)), 
            roc_auc_score(y_test, knn_pipe_scale.predict(X_test)), 
            roc_auc_score(y_test, rf_pipe_scale.predict(X_test))]
method_name = ['Logistic Regression', 'Decision Tree Classifier',
              'KNN Classifier', 'Random Forest Classifier']
cv_summary = pd.DataFrame({
    'method': method_name,
    'mean score': score_mean,
    'std score': score_std,
    'roc auc score': score_roc_auc
})
cv_summary

plot_roc_curve(tree_pipe_scale, X_test, y_test)

# Handling Imbalance

### UnderSampling

**RandomUnderSampler Model**

rus = RandomUnderSampler(random_state = 2727)
X_under, y_under = rus.fit_resample(X_train_val, y_train_val) 

logreg_pipe_scale_under = Pipeline([
    ('transformer', transformer_scale),
    ('rus', rus),
    ('logreg', logreg)
])

tree_pipe_scale_under = Pipeline([
    ('transformer', transformer_scale),
    ('rus', rus),
    ('tree', tree)
])

knn_pipe_scale_under = Pipeline([
    ('transformer', transformer_scale),
    ('rus', rus),
    ('knn', knn)
])

rf_pipe_scale_under = Pipeline([
    ('transformer', transformer_scale),
    ('rus', rus),
    ('rf', rf)
])

def model_evaluation(model, metric):
    skfold = StratifiedKFold(n_splits = 5)
    model_cv = cross_val_score(model, X_under, y_under, cv = skfold, scoring = metric)
    return model_cv  

logreg_scale_under_cv = model_evaluation(logreg_pipe_scale_under, 'roc_auc') 
tree_scale_under_cv = model_evaluation(tree_pipe_scale_under, 'roc_auc')
knn_pipe_under_cv = model_evaluation(knn_pipe_scale_under, 'roc_auc')
rf_pipe_under_cv = model_evaluation(rf_pipe_scale_under, 'roc_auc')

*Fitting Data*

for model in [logreg_pipe_scale_under, tree_pipe_scale_under, knn_pipe_scale_under, rf_pipe_scale_under]:
    model.fit(X_train_val, y_train_val)

*Summary*

score_mean = [logreg_scale_under_cv.mean(), tree_scale_under_cv.mean(), knn_pipe_under_cv.mean(),
              rf_pipe_under_cv.mean()]
score_std = [logreg_scale_under_cv.std(), tree_scale_under_cv.std(), knn_pipe_under_cv.std(),
             rf_pipe_under_cv.std()]
score_roc_auc = [roc_auc_score(y_test, logreg_pipe_scale_under.predict(X_test)),
            roc_auc_score(y_test, tree_pipe_scale_under.predict(X_test)), 
            roc_auc_score(y_test, knn_pipe_scale_under.predict(X_test)), 
            roc_auc_score(y_test, rf_pipe_scale_under.predict(X_test))]
method_name = ['Logistic Regression UnderSampling', 'Decision Tree Classifier UnderSampling',
              'KNN Classifier UnderSampling', 'Random Forest Classifier UnderSampling']
under_summary = pd.DataFrame({
    'method': method_name,
    'mean score': score_mean,
    'std score': score_std,
    'roc auc score': score_roc_auc
})
under_summary

plot_roc_curve(rf_pipe_scale_under, X_test, y_test)

**NearMiss Model**
- NearMiss adds some heuristic rules to select samples based on nearest neighbors algorithm.

nm = NearMiss(version = 1)

logreg_pipe_scale_nm = Pipeline([
    ('transformer', transformer_scale),
    ('nm', nm),
    ('logreg', logreg)
])

tree_pipe_scale_nm = Pipeline([
    ('transformer', transformer_scale),
    ('nm', nm),
    ('tree', tree)
])

knn_pipe_scale_nm = Pipeline([
    ('transformer', transformer_scale),
    ('nm', nm),
    ('knn', knn)
])

rf_pipe_scale_nm = Pipeline([
    ('transformer', transformer_scale),
    ('nm', nm),
    ('rf', rf)
])

def model_evaluation(model, metric):
    skfold = StratifiedKFold(n_splits = 5)
    model_cv = cross_val_score(model, X_train_val, y_train_val, cv = skfold, scoring = metric)
    return model_cv

logreg_scale_nm_cv = model_evaluation(logreg_pipe_scale_nm, 'roc_auc') 
tree_scale_nm_cv = model_evaluation(tree_pipe_scale_nm, 'roc_auc')
knn_pipe_nm_cv = model_evaluation(knn_pipe_scale_nm, 'roc_auc')
rf_pipe_nm_cv = model_evaluation(rf_pipe_scale_nm, 'roc_auc')

*Fitting Data*

for model in [logreg_pipe_scale_nm, tree_pipe_scale_nm, knn_pipe_scale_nm, rf_pipe_scale_nm]:
    model.fit(X_train_val, y_train_val)

*Summary*

score_mean = [logreg_scale_nm_cv.mean(), tree_scale_nm_cv.mean(), knn_pipe_nm_cv.mean(),
              rf_pipe_nm_cv.mean()]
score_std = [logreg_scale_nm_cv.std(), tree_scale_nm_cv.std(), knn_pipe_nm_cv.std(),
             rf_pipe_nm_cv.std()]
score_roc_auc = [roc_auc_score(y_test, logreg_pipe_scale_nm.predict(X_test)),
            roc_auc_score(y_test, tree_pipe_scale_nm.predict(X_test)), 
            roc_auc_score(y_test, knn_pipe_scale_nm.predict(X_test)), 
            roc_auc_score(y_test, rf_pipe_scale_nm.predict(X_test))]
method_name = ['Logistic Regression NearMiss', 'Decision Tree Classifier NearMiss',
              'KNN Classifier NearMiss', 'Random Forest Classifier NearMiss']
nm_summary = pd.DataFrame({
    'method': method_name,
    'mean score': score_mean,
    'std score': score_std,
    'roc auc score': score_roc_auc
})
nm_summary

plot_roc_curve(rf_pipe_scale_nm, X_test, y_test)

### OverSampling

**RandomOverSampler Model**

ros = RandomOverSampler(random_state = 2727)
X_over, y_over = ros.fit_resample(X_train_val, y_train_val)

logreg_pipe_scale_over = Pipeline([
    ('transformer', transformer_scale),
    ('ros', ros), 
    ('logreg', logreg)
])

tree_pipe_scale_over = Pipeline([
    ('transformer', transformer_scale),
    ('ros', ros), 
    ('tree', tree)
])

knn_pipe_scale_over = Pipeline([
    ('transformer', transformer_scale),
    ('ros', ros), 
    ('knn', knn)
])

rf_pipe_scale_over = Pipeline([
    ('transformer', transformer_scale),
    ('ros', ros),
    ('rf', rf)
])

def model_evaluation(model, metric):
    skfold = StratifiedKFold(n_splits = 5)
    model_cv = cross_val_score(model, X_over, y_over, cv = skfold, scoring = metric)
    return model_cv

logreg_scale_over_cv = model_evaluation(logreg_pipe_scale_over, 'roc_auc') 
tree_scale_over_cv = model_evaluation(tree_pipe_scale_over, 'roc_auc')
knn_pipe_over_cv = model_evaluation(knn_pipe_scale_over, 'roc_auc')
rf_pipe_over_cv = model_evaluation(rf_pipe_scale_over, 'roc_auc')

*Fitting Data*

for model in [logreg_pipe_scale_over, tree_pipe_scale_over, knn_pipe_scale_over, rf_pipe_scale_over]:
    model.fit(X_train_val, y_train_val)

score_mean = [logreg_scale_over_cv.mean(), tree_scale_over_cv.mean(), knn_pipe_over_cv.mean(),
              rf_pipe_over_cv.mean()]
score_std = [logreg_scale_over_cv.std(), tree_scale_over_cv.std(), knn_pipe_over_cv.std(),
             rf_pipe_over_cv.std()]
score_roc_auc = [roc_auc_score(y_test, logreg_pipe_scale_over.predict(X_test)),
            roc_auc_score(y_test, tree_pipe_scale_over.predict(X_test)), 
            roc_auc_score(y_test, knn_pipe_scale_over.predict(X_test)), 
            roc_auc_score(y_test, rf_pipe_scale_over.predict(X_test))]
method_name = ['Logistic Regression OverSampling', 'Decision Tree Classifier OverSampling',
              'KNN Classifier OverSampling', 'Random Forest Classifier OverSampling']
over_summary = pd.DataFrame({
    'method': method_name,
    'mean score': score_mean,
    'std score': score_std,
    'roc auc score': score_roc_auc
})
over_summary

plot_roc_curve(logreg_pipe_scale_over, X_test, y_test)

**SMOTE**

smote = SMOTE(random_state = 2727)

logreg_pipe_scale_smote = Pipeline([
    ('transformer', transformer_scale),
    ('smote', smote),
    ('logreg', logreg)
])

tree_pipe_scale_smote = Pipeline([
    ('transformer', transformer_scale),
    ('smote', smote),
    ('tree', tree)
])

knn_pipe_scale_smote = Pipeline([
    ('transformer', transformer_scale),
    ('smote', smote),
    ('knn', knn)
])

rf_pipe_scale_smote = Pipeline([
    ('transformer', transformer_scale),
    ('smote', smote),
    ('rf', rf)
])

def model_evaluation(model, metric):
    skfold = StratifiedKFold(n_splits = 5)
    model_cv = cross_val_score(model, X_train_val, y_train_val, cv = skfold, scoring = metric)
    return model_cv

logreg_scale_smote_cv = model_evaluation(logreg_pipe_scale_smote, 'roc_auc') 
tree_scale_smote_cv = model_evaluation(tree_pipe_scale_smote, 'roc_auc')
knn_pipe_smote_cv = model_evaluation(knn_pipe_scale_smote, 'roc_auc')
rf_pipe_smote_cv = model_evaluation(rf_pipe_scale_smote, 'roc_auc')

*Fitting Data*

for model in [logreg_pipe_scale_smote, tree_pipe_scale_smote, knn_pipe_scale_smote, rf_pipe_scale_smote]:
    model.fit(X_train_val, y_train_val)

*Summary*

score_mean = [logreg_scale_smote_cv.mean(), tree_scale_smote_cv.mean(), knn_pipe_smote_cv.mean(),
              rf_pipe_smote_cv.mean()]
score_std = [logreg_scale_smote_cv.std(), tree_scale_smote_cv.std(), knn_pipe_smote_cv.std(),
             rf_pipe_smote_cv.std()]
score_roc_auc = [roc_auc_score(y_test, logreg_pipe_scale_smote.predict(X_test)),
            roc_auc_score(y_test, tree_pipe_scale_smote.predict(X_test)), 
            roc_auc_score(y_test, knn_pipe_scale_smote.predict(X_test)), 
            roc_auc_score(y_test, rf_pipe_scale_smote.predict(X_test))]
method_name = ['Logistic Regression SMOTE', 'Decision Tree Classifier SMOTE',
              'KNN Classifier SMOTE', 'Random Forest Classifier SMOTE']
smote_summary = pd.DataFrame({
    'method': method_name,
    'mean score': score_mean,
    'std score': score_std,
    'roc auc score': score_roc_auc
})
smote_summary

plot_roc_curve(logreg_pipe_scale_smote, X_test, y_test)

# HyperParam Tuning

estimator = Pipeline([
    ('transformer', transformer_scale),
    ('rus', rus),
    ('model', rf)
])

hyperparam_space = {
    'model__n_estimators': [50, 100, 150, 200],
    'model__criterion': ['gini', 'entropy'],
    'model__max_depth': [2, 3, 5, 7, 10, 15, 20],
    'model__min_samples_leaf': [3, 7, 9, 13, 15, 21],
    'model__max_features': [20, 30, 40, 50]
}

random = RandomizedSearchCV(
                estimator,
                param_distributions = hyperparam_space,
                cv = StratifiedKFold(n_splits = 5),
                scoring = 'roc_auc',
                n_iter = 10,
                n_jobs = -1)

random.fit(X_train_val, y_train_val)

print('best score', random.best_score_)
print('best param', random.best_params_)

Comparing to all Handling Methods that I've been running, the best model is:
- RandomForestClassifier with UnderSampling
- Best Score: 0.88
- Best Estimator: 200
- Best Min Samples Leaf: 9
- Best Max Features: 20
- Best Max Depth: 15
- Best Criterion: entropy

# Before VS After Tuning

estimator.fit(X_train_val, y_train_val)
y_pred_estimator = estimator.predict(X_test)
roc_auc_estimator = roc_auc_score(y_test, y_pred_estimator)

random.best_estimator_.fit(X_train_val, y_train_val)
y_pred_random = random.best_estimator_.predict(X_test)
roc_auc_best_estimator = roc_auc_score(y_test, y_pred_random)

score_list = [roc_auc_estimator, roc_auc_best_estimator]
method_name = ['Random Forest Classifier UnderSampling Before', 'Random Forest Classifier UnderSampling After']
best_summary = pd.DataFrame({
    'method': method_name,
    'score': score_list
})
best_summary

# Using ML

import pickle

file_name = "data_hotel_booking_demand.csv"
pickle.dump(estimator, open(file_name, 'wb'))

loaded_model = pickle.load(open(file_name, 'rb'))

loaded_model.predict(X_test)

hotel_pred = pd.DataFrame({
    'country': ['PRT'],
    'market_segment': ['Online TA'],
    'previous_cancellations': ['0'],
    'booking_changes': ['0'],
    'deposit_type': ['No Deposit'],
    'days_in_waiting_list': ['0'],
    'customer_type': ['Contract'],
    'reserved_room_type': ['A'],
    'required_car_parking_spaces': ['0'],
    'total_of_special_requests': ['0'],
})

loaded_model.predict(hotel_pred)

loaded_model.predict_proba(hotel_pred)

