from sklearn.metrics import classification_report
import scikitplot as skplt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from IPython.display import Markdown, display
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from scipy import interp
import pandas as pd
import time 
import matplotlib.pyplot as plt
import numpy as np
import os

def class_report(y_true, y_pred, y_score=None, average='micro'):
    """
    calculate the classification report
    """
    
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    Parameters: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)#, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    #plt.savefig(filename)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t) # update the ylim(bottom, top) values
    plt.show()
    
def printmd(string):
    """
    Parameter:
        string: string to display
    """
    display(Markdown(string))

def get_metrics(classifier, X_train, y_train, X_test, y_test, target_names):
    """
    Parameters:
        classifier: type of classifier
        X_train:
        y_train:
        X_test:
        y_test:
        target_names: name the classifier.
    
    Return:
        plot important metrics
    """
    pred = classifier.predict(X_test)
    printmd('# <center> Start training classifier {} </center>'.format(target_names))

    print(classification_report(y_test, pred))
    print(cohen_kappa_score(y_test, pred))
    
    cm_analysis(y_test, pred, list(classifier.classes_), ymap=None, figsize=(25,15))
    
    y_probas = classifier.predict_proba(X_test)
    time.sleep(2)
    skplt.metrics.plot_roc(y_test, y_probas, figsize=(25,15))
    skplt.metrics.plot_precision_recall(y_test, y_probas, figsize=(25,15))
    
    report = class_report(y_test, pred, y_probas)
    report.insert(loc=0, column="algo_type", value=target_names)


    
    
    return report



def plot_the_feature_importance(feature_score:pd.DataFrame, model_name:str, path:str=None):
    """    
    Parameters:
        df: input feature_score
        model_name: 
        path: path to save
    Return:
        plot the feature importance

    """
    
    feature_score = feature_score.sort_values(by='Score', ascending=True, inplace=False, kind='quicksort', na_position='last')

    plt.rcParams["figure.figsize"] = (28,15)
    ax = feature_score.plot('Feature', 'Score', kind='barh', color='c')
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_title("Feature Importance using {}".format(model_name), fontsize = 24)
    if path is not None:
        plt.savefig(os.path.join(path,str(model_name)+"_feature_importance.pdf"))
    plt.show()

    

def drow_corr_map(data_in):
    """
    plot correlation matrix
    Parameters:
        data_in
    """
    fig,ax = plt.subplots(figsize=(35,25))
    df_corr = data_in.corr()
    sns.heatmap(df_corr, annot=True)
    b, t = ax.get_ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    ax.set_ylim(b, t) # update the ylim(bottom, top) values



def get_redundant_pairs(df:pd.DataFrame):
    '''
    Get diagonal and lower triangular pairs of the correlation matrix
    Parameter:
        df: Data Frame
    Return:
        set of features pairs
    '''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df:pd.DataFrame):
    """
    Generate the top corr features > 0.5
    Parameter:
        df: Data Frame
    Return:
        top corr features > 0.5
    """
    
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    top_corr_features = au_corr[au_corr>0.5]
    print("Top Absolute Correlations")
    print(top_corr_features)
    
    return top_corr_features