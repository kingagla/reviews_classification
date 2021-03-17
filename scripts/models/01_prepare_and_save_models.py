import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scripts.utils import create_dir
from scripts.settings import *


def prepare_for_learning(file_path, model_path, n_samples=5000, use_neutral=False):
    # load data
    rev_vec = pd.read_pickle(file_path)
    # remove neutral if not used
    if not use_neutral:
        rev_vec = rev_vec[rev_vec['Information'] != 'neu']
    # use only part of available data
    rev_vec = rev_vec.sample(n_samples)
    # save indices of training and validation set
    pickle.dump(rev_vec.index, open(learning_index_path, 'wb'))

    X, y = rev_vec[[col for col in rev_vec.columns if col.startswith('Vec')]], rev_vec['Information']
    le = LabelEncoder()
    le.fit(y.values.reshape(-1, 1))
    create_dir(os.path.dirname(model_path))
    pickle.dump(le, open(model_path, 'wb'))
    return rev_vec, X, y


def classification_report_to_excel(y_test, y_pred, excel_path):
    cr = classification_report(y_test, y_pred, output_dict=True)
    create_dir(os.path.dirname(excel_path))
    pd.DataFrame(cr).T.to_excel(excel_path)


def neural_network():
    model = Sequential()
    model.add(Dense(256, input_dim=1024, activation='relu', use_bias=True,
                    kernel_initializer='random_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', use_bias=True, kernel_initializer='random_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu', use_bias=True, kernel_initializer='random_normal'))
    model.add(Dense(1, activation='sigmoid', use_bias=True, kernel_initializer='random_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['acc'])
    return model


def fit_and_save_model(X_train, y_train, model, model_path, network=False):
    # create directory for model
    create_dir(os.path.dirname(model_path))

    if network:
        checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True)
        model.fit(X_train, y_train, epochs=150, batch_size=512, validation_split=0.2, callbacks=[checkpoint])
    else:
        model.fit(X_train, y_train)
        pickle.dump(model, open(model_path, 'wb'))


def main():
    rev_vec, X, y = prepare_for_learning(rev_path,
                                         os.path.join(model_dir, label_encoder_file),
                                         n_samples=5000,
                                         use_neutral=False)
    le_path = os.path.join(model_dir, label_encoder_file)
    le = pickle.load(open(le_path, 'rb'))
    y = le.transform(y)
    # learn random forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                min_samples_leaf=2,
                                class_weight='balanced', criterion='entropy')
    fit_and_save_model(X, y, rf, os.path.join(model_dir, random_forest_file), network=False)

    # use DBSCAN to find negative
    dbs = DBSCAN(eps=0.01, min_samples=2)
    pickle.dump(dbs, open(os.path.join(model_dir, dbscan_file), 'wb'))

    # use neural network
    network = neural_network()
    fit_and_save_model(X, y, network, os.path.join(model_dir, network_file), network=True)


if __name__ == '__main__':
    main()
