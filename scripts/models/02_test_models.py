import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from scripts.create_plots.confussion_matrix import plot_conf_matrix
from scripts.settings import *
from scripts.utils import mapping_from_clusters, classification_report_to_excel


def main():
    # load data
    rev_vec = pd.read_pickle(rev_path)

    # remove elements used for learning from df and neutral values
    ids_used = open(learning_index_path, 'rb')
    ids_used = pickle.load(ids_used)
    rev_vec = rev_vec[rev_vec['Information'] != 'neu']
    rev_vec.drop(ids_used, inplace=True)

    # get X and y
    X, y = rev_vec[[col for col in rev_vec.columns if col.startswith('Vec')]], rev_vec['Information']

    # label encoding
    le_loaded = pickle.load(open(os.path.join(model_dir, label_encoder_file), 'rb'))
    y_num = le_loaded.transform(y.values.reshape(-1, 1))

    # neural network moddel
    network_model = load_model(os.path.join(model_dir, network_file))
    y_network = network_model.predict_classes(X)

    # random forest
    rf = pickle.load(open(os.path.join(model_dir, random_forest_file), 'rb'))
    y_rf = rf.predict(X)

    # DBSCAN
    dbscan = pickle.load(open(os.path.join(model_dir, dbscan_file), 'rb'))
    y_dbscan = dbscan.fit_predict(X)
    vfunc = np.vectorize(mapping_from_clusters)
    y_dbscan = vfunc(y_dbscan)
    y_dbscan = le_loaded.transform(y_dbscan.reshape(-1, 1))

    y_preds = [y_network, y_rf, y_dbscan]
    filenames = ['network', 'rf', 'dbscan']

    # create classification report and confusion matrix
    for y_pred, filename in zip(y_preds, filenames):
        excel_file = os.path.join(files_dir, f'{filename}.xlsx')
        classification_report_to_excel(y_num, y_pred, excel_file)
        plot_file = os.path.join(plots_dir, 'confusion_matrix', f'{filename}.png')
        plot_conf_matrix(y_num, y_pred, plot_file)


if __name__ == '__main__':
    main()