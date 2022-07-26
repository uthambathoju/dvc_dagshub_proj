import dagshub
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
import joblib
import numpy as np


drop_cols = ["Name", "SibSp", "Parch", "Ticket"]
obj_col = "Survived"
train_df_path = "data/train.csv"
test_df_path = "data/test.csv"
sub_df_path = "data/sample_submission.csv"


def feature_engineering(raw_df):
    df = raw_df.copy()
    df["Cabin"] = df["Cabin"].apply(lambda x: x[:1] if x is not np.nan else np.nan)   
    df["Family"] = df["SibSp"] + df["Parch"]
    return df



def fit_model(train_X, train_y, random_state=42):
    clf = SGDClassifier(loss="modified_huber", random_state=random_state)
    clf.fit(train_X, train_y)
    return clf


def to_category(train_df, test_df):
    cat = ["Sex", "Cabin", "Embarked"]
    for col in cat:
        le = preprocessing.LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
    return train_df, test_df


def eval_model(clf, X, y):
    y_proba = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "average_precision": average_precision_score(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }

def submission(clf, X):
    sub = pd.read_csv(sub_df_path)
    sub[obj_col] = clf.predict(X)
    sub.to_csv("submission/submission.csv", index=False)   


def train():
    print("Loading data...")
    df_train = pd.read_csv(train_df_path, index_col="PassengerId")
    df_test = pd.read_csv(test_df_path, index_col="PassengerId")
    print("Engineering features...")
    y = df_train[obj_col]
    X = feature_engineering(df_train).drop(drop_cols + [obj_col], axis=1)
    test_df = feature_engineering(df_test).drop(drop_cols, axis=1)
    X, test_df = to_category(X, test_df)
    X.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    with dagshub.dagshub_logger() as logger:
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42, stratify=y
        )
        model = fit_model(X_train, y_train)
        print("Saving trained model...")
        joblib.dump(model, "models/model.joblib")
        logger.log_hyperparams(model_class=type(model).__name__)
        logger.log_hyperparams({"model": model.get_params()})
        print("Evaluating model...")
        train_metrics = eval_model(model, X_train, y_train)
        print("Train metrics:")
        print(train_metrics)
        logger.log_metrics({f"train__{k}": v for k, v in train_metrics.items()})
        test_metrics = eval_model(model, X_test, y_test)
        print("Test metrics:")
        print(test_metrics)
        logger.log_metrics({f"test__{k}": v for k, v in test_metrics.items()})
        print("Creating Submission File...")
        submission(model, test_df)


if __name__ == "__main__":
    train()        