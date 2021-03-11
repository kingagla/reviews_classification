import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_conf_matrix(y_test, y_pred, filename):
    # define confusion matrices (normalized along true or predicted label)
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



