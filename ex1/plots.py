import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import math
# utility function to plot ROC and AUC
def plot_roc_auc(labels, logsoftmax_data, title):
    probability = math.e ** logsoftmax_data
    fpr, tpr, thresholds = metrics.roc_curve(labels, probability)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ROC')
    display.plot()
    plt.title(title)
    plt.show()
