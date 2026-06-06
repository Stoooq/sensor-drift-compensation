from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_svm_classifier(n_components: int) -> Pipeline:
    pipeline = Pipeline([("pca", PCA(n_components)), ("clf", SVC(probability=True))])
    return pipeline


def build_rf_classifier(n_components: int) -> Pipeline:
    pipeline = Pipeline([("pca", PCA(n_components)), ("clf", RandomForestClassifier())])
    return pipeline
