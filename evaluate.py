import torch
import utils
from model import Model
import engine
import params
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt


def oneHot(arr):
    b = np.zeros((arr.size, arr.max() + 1))
    b[np.arange(arr.size), arr] = 1
    return b


def calc_roc_auc(all_labels, all_logits):
    attributes = ['Alzheimer\'s (-ve)', 'Alzheimer\'s (+ve)',
                  'Cancer (-ve)', 'Cancer (+ve)',
                  'Diabetes (-ve)', 'Diabetes (+ve)',
                  'Parkinson\'s (-ve)', 'Parkinson\'s (+ve)']
    all_labels = oneHot(all_labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0, len(attributes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='%s %g' % (attributes[i], roc_auc[i]))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.savefig(params.output_folder + "/ROC-AUC.pdf")
    plt.clf()
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]


def calculate_metrics(predictions, labels, probabilities):
    utils.log(f"Accuracy: {metrics.accuracy_score(labels, predictions)}")
    utils.log(f"Precision: {metrics.precision_score(labels, predictions, average='weighted')}")
    utils.log(f"Recall: {metrics.recall_score(labels, predictions, average='weighted')}")
    utils.log(f"F1-Score: {metrics.f1_score(labels, predictions, average='weighted')}")
    utils.log(f'ROC-AUC Score: {calc_roc_auc(np.array(labels), np.array(probabilities))}')
    utils.log(f"MCC Score: {metrics.matthews_corrcoef(labels, predictions)}")
    utils.log(f"Classification Report:\n{metrics.classification_report(labels, predictions, digits=4)}")
    
    np.set_printoptions(linewidth=np.inf)   # disable line wrap
    utils.log(metrics.confusion_matrix(labels, predictions))


def evaluate(model = None):
    _, _, test = utils.read_data()
    test_data_loader = utils.get_dataloader(test)
    device = utils.set_device()
    if model is None:
        model = Model()
        model = model.to(device)

    utils.log("##################################### Testing ############################################")
    model.load_state_dict(torch.load(params.output_folder + f"/Model.bin"))
    predictions, labels, probabilities = engine.test(test_data_loader, model, device)
    test['y_pred'] = predictions
    pred_test = test[['text', 'label', 'target', 'y_pred']]
    pred_test.to_csv(params.output_folder + '/Test Results.csv', index=False)
    
    calculate_metrics(predictions, labels, probabilities)

    del model, test_data_loader
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    utils.log("##################################### Task End ############################################")


if __name__ == "__main__":
    params.max_length = 128
    params.batch_size = 16
    params.learning_rate = 1e-05
    params.epochs = 15
    params.dropout = 0.4
    params.hidden_units = 64
    params.weight_decay = 1e-02
    params.epsilon = 1e-08
    params.pretrained_model = params.models.bert
    params.device = params.devices.cuda0
    params.data_path = "./data.txt"
    params.output_folder = f"./{params.pretrained_model.name}/"

    evaluate()
