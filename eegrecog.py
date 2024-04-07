import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
from pdpbox import pdp
from alibi.explainers import CounterFactual
from lime import lime_tabular
import lime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence 
from sklearn.calibration import CalibratedClassifierCV
from alibi.explainers import CounterFactualProto
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import learning_curve
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

brainwave_df = pd.read_csv('emotions.csv')

label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

def preprocess_inputs(df):
    df = df.copy()
    
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
    
    return X_train, X_test, y_train, y_test

# 모델과 데이터를 전달받아 PDP, LIME, Counterfactual Explanations을 수행하 결과를 출력하는 함수를 정의합니다.
def model_explanations(model, pdpmodel, X_train, X_test, y_test, sorted_index_desc, index=0):
    X_train_df = pd.DataFrame(X_train, columns=X_columns_name)
    X_test_df = pd.DataFrame(X_test, columns=X_columns_name)
    print("ICE")
    pdp_model = pdp.PDPIsolate(
        model=pdpmodel,
        df=X_test_df,
        model_features=X_test_df.columns.tolist(),
        feature=X_test_df.columns[sorted_index_desc[0]],
        feature_name=X_test_df.columns[sorted_index_desc[0]],
        n_classes=3
    )
    fig, _ = pdp_model.plot(plot_lines=True)
    fig.show()
    
    print("LIME Explanations")
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=X_train_df.columns,
        class_names=['Negative', 'Neutral', 'Positive'],
        categorical_features=X_train_df.columns,
        mode='classification'
    )
    exp = explainer.explain_instance(
        data_row=X_train_df.iloc[100],
        predict_fn=model.predict_proba,
        num_samples=10000,
         num_features=15
    )
    exp.show_in_notebook(show_table=True, show_all=False)

    # print("Counterfactual Explanations")
    # counterfactual = CounterFactualProto(model.predict_proba, X_train_df.shape[1], C=100, optimizer="adam", tol=1e-5)
    # counterfactual_instance = counterfactual.explain(X_test_df.iloc[index].values.reshape(1, -1))
    # print("Original instance: ")
    # print("Target label: ", y_test.iloc[index])
    # print("Predicted label: ", np.argmax(model.predict(X_test_df.iloc[index].values.reshape(1, -1))))

    # # Print counterfactual instance
    # print("Counterfactual instance: ")
    # print("Counterfactual label: ", counterfactual_instance.cf['class'])

X_train, X_test, y_train, y_test = preprocess_inputs(brainwave_df)
X_columns_name = X_test.columns

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def SVM():
    # LinearSVC
    svc = LinearSVC(max_iter=5000, C=0.005, penalty='l1', dual=False)
    svc.fit(X_train_scaled, y_train)
    svc_score = svc.score(X_test_scaled, y_test)

    # LinearSVC coefficients
    svc_coef = svc.coef_
    svc_coef_df = pd.DataFrame({'feature': X_train.columns, 'importance': svc_coef[0]})
    print("LinearSVC coefficients:")
    print(svc_coef_df)
    
    calibrated_svc = CalibratedClassifierCV(svc, cv='prefit')
    calibrated_svc.fit(X_train_scaled, y_train)
    calibrated_svc_score = calibrated_svc.score(X_test_scaled, y_test)
    
    return svc, calibrated_svc, svc_score, calibrated_svc_score, svc_coef_df

def RandomForest():
    # RandomForestClassifier
    rf = RandomForestClassifier(max_depth=4, min_samples_leaf=10)  # Limit the depth of the trees
    rf.fit(X_train_scaled, y_train)
    rf_score = rf.score(X_test_scaled, y_test)

    # RandomForest feature importance
    rf_importances = rf.feature_importances_
    rf_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': rf_importances})
    print("RandomForest feature importance:")
    print(rf_importance_df.sort_values('importance', ascending=False))

    return rf, rf_score, rf_importance_df


def XGBC():
    # XGBoost
    xgbc = xgb.XGBClassifier(max_depth=2, learning_rate=0.2, n_estimators=30, subsample=0.5, 
                             reg_alpha=0.1, reg_lambda=1.0,
                             use_label_encoder=False, objective='multi:softprob', 
                             eval_metric='mlogloss', tree_method='gpu_hist')
    xgbc.fit(X_train_scaled, y_train)
    xgbc_score = xgbc.score(X_test_scaled, y_test)

    # XGBoost feature importance
    xgbc_importances = xgbc.feature_importances_
    xgbc_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': xgbc_importances})
    print("XGBoost feature importance:")
    print(xgbc_importance_df.sort_values('importance', ascending=False))
    return xgbc, xgbc_score, xgbc_importance_df

def sort_index(data):
    sorted_index_desc = sorted(len(data), key=lambda i: data[i], reverse=True)
    return sorted_index_desc

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'][:15], y=fi_df['feature_names'][:15])
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')

svc, calibrated_svc, svc_score, calibrated_svc_score, svc_coef_df = SVM()
rf, rf_score, rf_importance_df = RandomForest()
xgbc, xgbc_score, xgbc_importance_df = XGBC()


# 출력
# 출력
print(f"SVM: {svc_score}")
print(f"SVM+Calibration: {calibrated_svc_score}")
print(f"Random Forest: {rf_score}")
print(f"Gradient Boosting: {xgbc_score}")


# SHAP values for RandomForest
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")

# 각 모델에 대해 설명 결과 출력
model_examples = [("SVM", calibrated_svc, svc, svc_coef_df), ("Random Forest", rf, rf, rf_importance_df), ("Gradient Boosting", xgbc, xgbc, xgbc_importance_df)]

for model_name, model, pdpmodel, coef in model_examples:
    print(f"\n{model_name}:")
    sorted_index_desc = sort_index(coef['coefficient'])
    model_explanations(model, pdpmodel, X_train_scaled, X_test_scaled, y_test, sorted_index_desc)