import os
import morfeusz2
import pandas as pd
from sklearn.metrics import classification_report


def lemmatize_text(text):
    if isinstance(text, str):
        text = text.split()
    morf = morfeusz2.Morfeusz(expand_dag=True, expand_tags=True)
    text_new = []
    for word in text:
        w = morf.analyse(word)[0][0][1].split(':')[0]
        text_new.append(w)
    return " ".join(text_new)


def create_dir(directory):
    if not os.path.isdir(directory):
        _path = os.path.abspath(directory).split('\\')
        for i in range(1, len(_path) + 1):
            current_dir = "//".join(_path[:i])
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)


def mapping_from_clusters(x):
    if x == -1:
        return 'neg'
    else:
        return 'pos'


def classification_report_to_excel(y_test, y_pred, filename):
    cr = classification_report(y_test, y_pred, output_dict=True, target_names=['Negative', 'Positive'])
    pd.DataFrame(cr).T.to_excel(filename)
