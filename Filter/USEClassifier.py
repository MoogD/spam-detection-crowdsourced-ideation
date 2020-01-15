from os import listdir
from os.path import isfile, join

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

import variables
from Helper import importDataHelper


def load_data(challenge):
    print("USE loaded")
    data = {"DESCRIPTION": [], "SPAM": []}
    if challenge == "all":
        idealist = []
        for file in listdir(variables.ideadbpath):
            if isfile(join(variables.ideadbpath, file)):
                idealist += list(importDataHelper.readcsvdata(join(variables.ideadbpath, file)))
    else:
        idealist = list(importDataHelper.readcsvdata(variables.ideadbpath + challenge + ".csv"))
    for idea in idealist:
        data["DESCRIPTION"].append(idea["DESCRIPTION"])
        if "unusable" in idea.get("STATUS", ""):
            data["SPAM"].append(1)
        elif "usable" in idea.get("STATUS", ""):
            data["SPAM"].append(0)
        elif "spam" in idea.get("SPAM", ""):
            data["SPAM"].append(1)
        else:
            data["SPAM"].append(0)
    return pd.DataFrame(data)

def train_classifier_idealist_use(X_train, path=None):
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        X_train, X_train["Spam"], num_epochs=None, shuffle=True)

    embedded_text_feature_column = hub.text_embedding_column(
        key="DESCRIPTION",
        module_spec="https://tfhub.dev/google/universal-sentence-encoder/2")
    estimator = tf.compat.v1.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003))
    estimator.train(input_fn=train_input_fn, steps=100)
    if path is not None:
        try:
            serving_input_fn = tf.compat.v1.estimator.export.build_parsing_serving_input_receiver_fn(
                tf.feature_column.make_parse_example_spec([embedded_text_feature_column]))
            estimator.export_saved_model(
                path, serving_input_fn)
        except:
            print("Could not save USE for ", path)
    return estimator

def train_classifier_idealist(X_train, path=None):
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        X_train, X_train["Spam"], num_epochs=None, shuffle=True)

    embedded_text_feature_column = hub.text_embedding_column(
        key="DESCRIPTION",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    estimator = tf.compat.v1.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003))
    estimator.train(input_fn=train_input_fn, steps=100)
    if path is not None:
        try:
            serving_input_fn = tf.compat.v1.estimator.export.build_parsing_serving_input_receiver_fn(
                tf.feature_column.make_parse_example_spec([embedded_text_feature_column]))
            estimator.export_saved_model(
                path, serving_input_fn)
        except:
            print("Could not save USE for ", path)
    return estimator

def train_classifier_plot(challenge):
    X_train = load_data(challenge)
    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        X_train, X_train["SPAM"], num_epochs=None, shuffle=True)

    fpr_fabric = dict()
    tpr_fabric = dict()
    roc_auc_fabric = dict()
    fpr_bionic = dict()
    tpr_bionic = dict()
    roc_auc_bionic = dict()
    fpr_tco = dict()
    tpr_tco = dict()
    roc_auc_tco = dict()

    embedded_text_feature_column = hub.text_embedding_column(
        key="DESCRIPTION",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    estimator = tf.compat.v1.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.compat.v1.train.AdagradOptimizer(learning_rate=0.003))
    estimator.train(input_fn=train_input_fn, steps=100)

    return estimator

def classify(estimator, idea):
    res = list(estimator.predict(tf.compat.v1.estimator.inputs.pandas_input_fn(
        pd.DataFrame(idea, index=[0]), num_epochs=1, shuffle=False)))[0]
    return res["class_ids"][0], res["probabilities"][res["class_ids"][0]]
