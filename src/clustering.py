import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from utils_metrics_plot import get_metrics
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


def make_clustering(data_in: pd.DataFrame, algo_cluster, path_to_save: str = '', info: bool = False):
    """
    Parameters:
        data_in: input dataset
        algo_cluster: algo that is using for clustering
        info: True if additional info should be print.

    TIP. info - need time.

    Return:
        labels: labels for cluster
    """

    cmap = 'Spectral'
    data_transform = data_in
    if algo_cluster == DBSCAN:
        clustering = algo_cluster.fit(data_transform)
        labels = clustering.labels_

    else:
        labels = algo_cluster.fit_predict(data_transform)

    fig, ax = plt.subplots(figsize=(20, 20))
    for g in np.unique(labels):
        ix = np.where(labels == g)
        ax.scatter(data_in[ix, 0], data_in[ix, 1], label=g, s=10, cmap=cmap)
    ax.legend(prop={'size': 20})
    if path_to_save:
        plt.savefig(path_to_save)
    plt.show()

    if info:
        print("clusters_dereils", pd.Series(labels).value_counts().T)
        n_clusters = len(pd.Series(labels).unique())
        silhouette_avg = silhouette_score(data_in, labels)
        print("For n_clusters = {}. \nThe average silhouette_score is: {}".format(n_clusters, silhouette_avg))

    return labels


def train_model_to_predict_clusters(check_values: np.ndarray, embeding_values: pd.DataFrame):
    """
    Train the model to identify the labels
    Parameters:
        check_values: df that will be splitted
        embeding_values: target column
        preprocessing: true if check_values should be preprocessed
    Return:
        rf: the trained model that can be classified
    """

    matrix_for_prediction = check_values

    X_tr_cluster, X_test_cluster, y_tr_cluster, y_test_cluster = train_test_split(matrix_for_prediction,
                                                                                  embeding_values, test_size=0.4,
                                                                                  random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_tr_cluster, y_tr_cluster)

    get_metrics(classifier=rf, X_train=X_tr_cluster, y_train=y_tr_cluster, X_test=X_test_cluster, y_test=y_test_cluster,
                target_names="random_forest")

    return rf

