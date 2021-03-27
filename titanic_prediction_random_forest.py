import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("train.csv")
df.head()
df.shape


#############################################
# 1. FEATURE ENGINEERING
#############################################

df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int')
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'


df.head()
df.shape

df.columns = [col.upper() for col in df.columns]

#############################################
# 2. AYKIRI GOZLEM
#############################################

num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "PASSENGERID"]


df.head()

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df, col))


from helpers.data_prep import replace_with_thresholds

for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))

from helpers.eda import check_df
check_df(df)


#############################################
# 3. EKSIK GOZLEM
#############################################
check_df(df)

from helpers.data_prep import missing_values_table
missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)
missing_values_table(df)

remove_vars = ["TICKET", "NAME"]
df.drop(remove_vars, inplace=True, axis=1)
df.head()
missing_values_table(df)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))


missing_values_table(df)

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


#############################################
# 4. LABEL ENCODING
#############################################

df.head()
df.shape

binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

from helpers.data_prep import label_encoder


for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. ONE-HOT ENCODING
#############################################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


#############################################
# 7. MODEL
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED","NEW_TITLE"], axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

#RANDOM FOREST
rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)

# train hatası
y_pred = rf_model.predict(X_train)
accuracy=np.sqrt(accuracy_score(y_train, y_pred))
print("Accuracy:%.2f%%" % (accuracy * 100.0))

# test hatası
y_pred = rf_model.predict(X_test)
accuracy=np.sqrt(accuracy_score(y_test, y_pred))
print("Accuracy:%.2f%%" % (accuracy * 100.0))


rf_params = {"max_depth": [5, 7, 9 ],
             "max_features": [3,5,15],
             "n_estimators": [100, 200, 300],
             "min_samples_split": [2, 5, 12]}


rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model.best_params_


#######################################
# Final Model
#######################################

rf_tuned = RandomForestClassifier(**rf_cv_model.best_params_).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:%.2f%%" % (accuracy * 100.0)) #82.68

#######################################
# Feature Importance
#######################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_tuned, X_train)
