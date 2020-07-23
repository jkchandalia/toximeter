from sklearn import metrics
import matplotlib.pyplot as plt


def run_metrics(predictions, predictions_prob, target, output_path=None, visualize=False):
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions_prob)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(target, predictions_prob)
    auc = metrics.auc(recall, precision)
    average_precision = metrics.average_precision_score(target, predictions)

    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    accuracy = metrics.accuracy_score(target, predictions)
    print(metrics.confusion_matrix(target, predictions, labels=[0,1]))
    print("Accuracy Score: {0:0.2f}".format(accuracy))
    if visualize:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC curve, AUC: {0:0.2f}'.format(roc_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        if output_path:
          plt.savefig(output_path+'\ROC.png')
        
        plt.figure()
        plt.plot(recall, precision)
        plt.title('Precision-Recall curve, AUC: {0:0.2f}'.format(auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if output_path:
          plt.savefig(output_path+'\Precision_Recall.png')
    return fpr, tpr, precision

