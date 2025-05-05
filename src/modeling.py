from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def train_model(X_train, y_train):
    model =  GradientBoostingClassifier(verbose=1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_model_xgboost(X_train_full, y_train_full):

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42
    )
    model = xgb.XGBClassifier(
        tree_method='gpu_hist',  
        predictor='gpu_predictor',
        n_estimators=500,
        eval_metric='auc',
        use_label_encoder=False
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True  
    )
    return model
