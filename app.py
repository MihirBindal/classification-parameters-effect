import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer

st.title("Effects of parameters and scaling on different classification algorithms")
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)
scaling = st.sidebar.selectbox(
    'Select Scaling',
    ("None", "Standard Scaler", "MaxAbsolute Scaler", "MinMax Scaler", "PowerTransformer", "Normalization")
)
st.write(f"## {dataset_name} Dataset")
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'LogisticRegression', 'SVM', 'Random Forest', 'xgboost')
)


def get_datasets(name):
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    if scaling == "Standard Scaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "Max Absolute Scaler":
        scaler = MaxAbsScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "MinMax Scaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "Normalization":
        scaler = Normalizer()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "PowerTransformer":
        scaler = PowerTransformer()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    y = data.target
    return X_scaled, y


X, y = get_datasets(dataset_name)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        kernel = st.sidebar.selectbox('Select kernel', ('linear', 'poly', 'rbf', 'sigmoid'))
        params['kernel'] = kernel
        degree = 3
        if kernel == 'poly':
            degree = st.sidebar.slider('degree', 3, 10)
            params['degree'] = degree
    elif clf_name == 'KNN':
        n_neighbors = st.sidebar.slider('n_neighbors', 1, 15)
        params['n_neighbors'] = n_neighbors
    elif clf_name == 'LogisticRegression':
        C = st.sidebar.slider('C', 1.0, 150.0)
        params['C'] = C
    elif clf_name == 'xgboost':
        eta = st.sidebar.slider('eta', 0.0, 1.0)
        params['eta'] = eta
        min_child_weight = st.sidebar.slider('min_child_weight', 0, 50)
        params['min_child_weight'] = min_child_weight
        gamma = st.sidebar.slider('C', 0, 50)
        params['gamma'] = gamma
        max_depth = st.sidebar.slider('max_depth', 6, 50)
        params['max_depth'] = max_depth
        alpha = st.sidebar.slider('alpha', 0, 15)
        params['alpha'] = alpha
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
        min_samples_split = st.sidebar.slider('min_samples_split', 2, 50)
        params['min_samples_split'] = min_samples_split
        min_samples_leaf = st.sidebar.slider('min_samples_leaf', 1, 50)
        params['min_samples_leaf'] = min_samples_leaf
        max_leaf_nodes = st.sidebar.slider('max_leaf_nodes', 2, 15)
        params['max_leaf_nodes'] = max_leaf_nodes
    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    classifier = None
    if clf_name == 'SVM':
        if params['kernel'] == 'poly':
            classifier = SVC(C=params['C'], kernel=params['kernel'], degree=params['degree'])
        else:
            classifier = SVC(C=params['C'], kernel=params['kernel'])
    elif clf_name == 'KNN':
            classifier = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
    elif clf_name == 'LogisticRegression':
        classifier = LogisticRegression(C=params['C'])
    elif clf_name == 'xgboost':
        classifier = XGBClassifier(max_depth=params['max_depth'], gamma=params['gamma'],
                                   min_child_weight=params['min_child_weight'], alpha=params['alpha'],
                                   eta=params['eta'])
    else:
        classifier = classifier = RandomForestClassifier(n_estimators=params['n_estimators'],
                                                         max_depth=params['max_depth'], min_samples_split=params['min_samples_split']
                                                         , min_samples_leaf=params['min_samples_leaf'],
                                                         max_leaf_nodes=params['max_leaf_nodes'], random_state=1234)
    return classifier


clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_final = clf.predict(X)
acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y_final, alpha=0.7,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
st.pyplot()
