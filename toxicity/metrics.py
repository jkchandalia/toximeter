

def run_metrics(predictions, predictions_prob, target, visualize=True):
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions_prob)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds = metrics.precision_recall_curve(target, predictions_prob)
    average_precision = metrics.average_precision_score(yvalid, pred)
    #average_recall = metrics.recall_score(yvalid, pred)
    print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    accuracy = metrics.accuracy_score(yvalid, pred)
    print(metrics.confusion_matrix(yvalid, pred, labels=[0,1]))
    print("Accuracy Score: {0:0.2f}".format(accuracy))
    if visualize:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC curve, AUC: {0:0.2f}'.format(roc_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        plt.show()
        
        plt.figure()
        plt.plot(recall, precision)
        plt.title('Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
        #disp = metrics.plot_precision_recall_curve(nb_classifier, count_valid, yvalid)
        #disp.ax_.set_title('2-class Precision-Recall curve: '
                   #'AP={0:0.2f}'.format(average_precision))

