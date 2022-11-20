# import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools
import random, pickle
import pprint
import numpy as np
import pandas as pd
import warnings

# supress warning
warnings.filterwarnings('ignore')

# utility functions

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

# other types of preprocessing
# - image : 8x8 : resize 16x16, 32x32, 4x4 : flatteing
# - normalize data: mean normalization: [x - mean(X)]
#                 - min-max normalization
# - smoothing the image: blur on the image

def data_viz(dataset):
    # PART: sanity check visualization of the data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

# PART: Sanity check of predictions
def pred_image_viz(x_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

# PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model

def train_dev_test_split(data, label, train_frac, dev_frac):

    dev_test_frac = 1 - train_frac
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size=dev_test_frac, shuffle=True
    )
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac) / dev_test_frac, shuffle=True
    )

    return x_train, y_train, x_dev, y_dev, x_test, y_test


def random_split(X, y):
    assert hasattr(y, '__iter__') and not hasattr(y[0], '__iter__')
    new_idxs = [i for i in range(len(y))]
    random.shuffle(new_idxs)
    return X[new_idxs], y[new_idxs]

def h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None
    best_clf_rep = None
    best_conf_mat = None
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        # PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        # PART: Train model
        # 2.a train the model
        # Learn the digits on the train subset
        clf.fit(x_train, y_train)

        # print(cur_h_params)
        # PART: get dev set predictions
        predicted_dev = clf.predict(x_dev)

        # 2.b compute the accuracy on the validation set
        cur_metric = metric(y_pred=predicted_dev, y_true=y_dev)
        clf_rep = metrics.classification_report(y_dev, predicted_dev, output_dict=True)
        conf_mat = metrics.confusion_matrix(y_dev, predicted_dev)
        # print("clf_rep: ", clf_rep)
        # 3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest.
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            best_clf_rep = clf_rep
            best_conf_mat = conf_mat

            # print("Found new best metric with :" + str(cur_h_params))
            # print("New best val metric:" + str(cur_metric))
    return best_model, best_metric, best_h_params, best_clf_rep, best_conf_mat

def prepare_hparams(params: dict):
    keys, values = zip(*params.items())
    for row in itertools.product(*values):
        yield dict(zip(keys, row))

def run_splits(data, label, train_frac, dev_frac, parameters, metric):
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac)
    results = {}
    save_model_flag = False
    for model_name in parameters:
        model = parameters[model_name]['model']()
        params = parameters[model_name]['params']
        h_param_comb = [comb for comb in prepare_hparams(params)]
        best_model, best_metric, best_h_params, best_clf_rep, conf_mat = h_param_tuning(h_param_comb, model, x_train, y_train, x_dev, y_dev, metric)
        results[model_name] = best_metric, best_h_params, best_clf_rep, conf_mat
        if not save_model_flag:
            # save
            with open('model.pkl','wb') as f:
                pickle.dump(best_model,f)
                save_model_flag = True
    return results

def run_multiple_models(data, label, parameters, no_of_splits, train_frac, dev_frac, test_frac, metric):
    assert train_frac + dev_frac + test_frac == 1.
    results = {}
    for split in range(no_of_splits):
        data, label = random_split(data, label)
        res = run_splits(data, label, train_frac, dev_frac, parameters, metric)
        results[split] = res
    return results

# Analysis functions

def compute_TP_FP_TN_FN(confusion_matrix):
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=['class_'+str(i) for i in range(len(confusion_matrix[0]))], index=['class_'+str(i) for i in range(len(confusion_matrix[0]))])
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.values.sum() - (FP + FN + TP)
    FP, FN, TP, TN = list(FP), list(FN), list(TP), list(TN)
    # res = {'true_positive': TP, 'false_positive': FP, 'true_negative': TN, 'false_negative': FN}
    return FP, FN, TP, TN

def analyse_result(results):
    # print(results[list(results.keys())[0]]['svc'])
    svc_res = {ky: {'acc': results[ky]['svc'][0], 'best_h_params': results[ky]['svc'][1], 'best_clf_report': results[ky]['svc'][2], 'conf_mat': results[ky]['svc'][3]} for ky in results}
    svc_acc = [svc_res[ky]['acc'] for ky in svc_res]
    dst_res = {ky: {'acc': results[ky]['decision_tree'][0], 'best_h_params': results[ky]['decision_tree'][1], 'best_clf_report': results[ky]['decision_tree'][2], 'conf_mat': results[ky]['decision_tree'][3]} for ky in results}
    dst_acc = [dst_res[ky]['acc'] for ky in dst_res]
    result = {'svm': {ky: svc_res[ky]['acc'] for ky in svc_res}, 'decision_tree': {ky: dst_res[ky]['acc'] for ky in dst_res}}
    result['svm']['mean'], result['svm']['std'] = np.mean(svc_acc), np.std(svc_acc)
    result['decision_tree']['mean'], result['decision_tree']['std'] = np.mean(dst_acc), np.std(dst_acc)
    df = pd.DataFrame(result)
    svc_metrics = compute_TP_FP_TN_FN(svc_res[1]['conf_mat'])
    dst_metrics = compute_TP_FP_TN_FN(dst_res[1]['conf_mat'])
    analyzed_res = {'TP_SVM': svc_metrics[2], 'TP_DT': dst_metrics[2], 'FN_SVM': svc_metrics[1], 'FN_DT': dst_metrics[1], 'FP_SVM': svc_metrics[0], 'FP_DT': dst_metrics[0], 'TN_SVM': svc_metrics[3], 'TN_DT': dst_metrics[3]}
    analyzed_res = pd.DataFrame(analyzed_res, index=['class'+str(i) for i in range(len(svc_metrics[0]))])
    return df, analyzed_res

if __name__ == '__main__':
    # train model section
    parameters = {
    'svc': {'model': SVC, 'params': {'kernel': ['rbf', 'poly', 'linear'], 'gamma': [0.01, 0.005, 0.001], 'C': [0.1, 0.2, 0.5, 1]}},
    'decision_tree': {'model': DecisionTreeClassifier, 'params': {'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt']}}
    }
    no_of_splits = 5
    train_frac, dev_frac, test_frac = 0.8, 0.1 , 0.1
    metric=metrics.accuracy_score

    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    results = run_multiple_models(data, label, parameters, no_of_splits, train_frac, dev_frac, test_frac)

    # result analysis section
    analysed_result = analyse_result(results)
    print(analysed_result[0])
    print(analysed_result[1])
