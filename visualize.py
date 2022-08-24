import matplotlib.pyplot as plt
import params


def save_acc_curves(history):
    fig = plt.figure(figsize=[6, 6])
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="lower right")
    plt.ylim([0, 1])
    plt.savefig(params.output_folder + "/Accuracy.svg")
    plt.close(fig)


def save_loss_curves(history):
    fig = plt.figure(figsize=[6, 6])
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc="upper right")
    plt.ylim([0, 1])
    plt.savefig(params.output_folder + "/Loss.svg")
    plt.close(fig)
