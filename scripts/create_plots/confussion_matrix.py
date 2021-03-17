import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scripts.utils import create_dir


def plot_conf_matrix(y_test, y_pred, filename):
    # create directory if doesn't exist
    directory = os.path.dirname(filename)
    if not os.path.isdir(directory):
        create_dir(directory)
    # define confusion mat1rices (normalized along true or predicted label)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm2 = confusion_matrix(y_test, y_pred, normalize='pred')

    # change cm to df
    df_cm = pd.DataFrame(cm, columns=['Negative', 'Positive'], index=['Negative', 'Positive'])
    df_cm2 = pd.DataFrame(cm2, columns=['Negative', 'Positive'], index=['Negative', 'Positive'])

    # plot as heatmap
    plt.figure(figsize=(15, 7))
    plt.subplot(121)
    sns.heatmap(df_cm, annot=True, cmap='Blues', cbar=False)
    plt.title('Normalizes confusion matrix over the true labels')
    plt.subplot(122)
    sns.heatmap(df_cm2, annot=True, cmap='Blues', cbar=False)
    plt.title('Normalizes confusion matrix over the predicted labels')
    plt.savefig(filename)
