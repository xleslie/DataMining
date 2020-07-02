import numpy as np
from numpy import *
import os
import pandas as pd
from pandas import DataFrame
from pyod.utils import evaluate_print, precision_n_scores
from pyod.utils.example import visualize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
# from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF
from sklearn.utils import *

pd.set_option('display.max_column',100)
n_clusters=8
classifiers={
    'abod':ABOD(n_neighbors=15),
    'knn':KNN(),
    # 'cblof':CBLOF(n_clusters=n_clusters),
    'fg':FeatureBagging(),
    'hbos':HBOS(),
    'if':IForest(),
    'lof':LOF()
}
dict={'csvname':[],
      'roc_abod_train':[],
      'roc_abod_test':[],
      'prn_abod_train':[],
      'prn_abod_test':[],
      'roc_knn_train':[],
      'roc_knn_test':[],
      'prn_knn_train':[],
      'prn_knn_test':[],
      # 'roc_cblof_train':[],
      # 'roc_cblof_test':[],
      # 'prn_cblof_train':[],
      # 'prn_cblof_test':[],
      'roc_fg_train':[],
      'roc_fg_test':[],
      'prn_fg_train':[],
      'prn_fg_test':[],
      'roc_hbos_train':[],
      'roc_hbos_test':[],
      'prn_hbos_train':[],
      'prn_hbos_test':[],
      'roc_if_train':[],
      'roc_if_test':[],
      'prn_if_train':[],
      'prn_if_test':[],
      'roc_lof_train':[],
      'roc_lof_test':[],
      'prn_lof_train':[],
      'prn_lof_test':[]}
draw={'y_abod_test':[],
      'y_abod_pred':[],
      'y_knn_test':[],
      'y_knn_pred':[],
      # 'y_cblof_pred':[],
      # 'y_cblof_test':[],
      'y_fg_test':[],
      'y_fg_pred':[],
      'y_hbos_test':[],
      'y_hbos_pred':[],
      'y_if_test':[],
      'y_if_pred':[],
      'y_lof_test':[],
      'y_lof_pred':[]}
def detection(path,file):
    data = pd.read_csv(path)
    data.drop(['point.id', 'motherset', 'origin'], axis=1, inplace=True)
    x = data.drop(['ground.truth'], axis=1, inplace=False)
    ground_truth_mapping = {'nominal': 0, 'anomaly': 1}
    y = data['ground.truth'].map(ground_truth_mapping)
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.3)
    train_count=train_Y.value_counts()
    test_count=test_Y.value_counts()
    if len(train_count.values) == 1 or len(test_count.values)==1:
        return
    # elif test_count.values[1]<=10 or train_count.values[1]<=10:
        n_clusters=3
    dict['csvname'].append(file)
    print("\nProprecess file:"+file)
    for i,(clf_name,clf) in enumerate(classifiers.items()):
        clf.fit(train_X)
        y_train_scores=clf.decision_scores_
        y_test_scores=clf.decision_function(test_X)
        train_Y = column_or_1d(train_Y)
        y_train_scores = column_or_1d(y_train_scores)
        check_consistent_length(train_Y, y_train_scores)
        roc_train = np.round(roc_auc_score(train_Y, y_train_scores), decimals=4)
        prn_train = np.round(precision_n_scores(train_Y, y_train_scores), decimals=4)
        test_Y=column_or_1d(test_Y)
        y_test_scores=column_or_1d(y_test_scores)
        check_consistent_length(test_Y,y_test_scores)

        roc_test = np.round(roc_auc_score(test_Y, y_test_scores), decimals=4)
        prn_test = np.round(precision_n_scores(test_Y, y_test_scores), decimals=4)
        dict['roc_'+clf_name+'_train'].append(roc_train)
        dict['roc_'+clf_name+'_test'].append(roc_test)
        dict['prn_'+clf_name+'_train'].append(prn_train)
        dict['prn_'+clf_name+'_test'].append(prn_test)
        draw['y_'+clf_name+'_test'].extend(test_Y)
        draw['y_'+clf_name+'_pred'].extend(y_test_scores)

def plot_roc(draw):
    false_positive_rate_abod,true_positive_rate_abod,thresholds_abod=roc_curve(draw['y_abod_test'], draw['y_abod_pred'])
    false_positive_rate_knn, true_positive_rate_knn, thresholds_knn = roc_curve(draw['y_knn_test'],
                                                                                   draw['y_knn_pred'])
    false_positive_rate_lof, true_positive_rate_lof, thresholds_lof = roc_curve(draw['y_lof_test'],
                                                                                   draw['y_lof_pred'])
    false_positive_rate_if, true_positive_rate_if, thresholds_if = roc_curve(draw['y_if_test'],
                                                                                   draw['y_if_pred'])
    false_positive_rate_fg, true_positive_rate_fg, thresholds_fg = roc_curve(draw['y_fg_test'],
                                                                                   draw['y_fg_pred'])
    false_positive_rate_hbos, true_positive_rate_hbos, thresholds_hbos = roc_curve(draw['y_hbos_test'],
                                                                                   draw['y_hbos_pred'])
    roc_auc_abod=auc(false_positive_rate_abod, true_positive_rate_abod)
    roc_auc_knn = auc(false_positive_rate_knn, true_positive_rate_knn)
    roc_auc_lof = auc(false_positive_rate_lof, true_positive_rate_lof)
    roc_auc_if = auc(false_positive_rate_if, true_positive_rate_if)
    roc_auc_fg = auc(false_positive_rate_fg, true_positive_rate_fg)
    roc_auc_hbos = auc(false_positive_rate_hbos, true_positive_rate_hbos)
    plt.title('ROC')
    plt.plot(false_positive_rate_abod, true_positive_rate_abod,label='ABOD:AUC = %0.4f'% roc_auc_abod)
    plt.plot(false_positive_rate_knn, true_positive_rate_knn, label='KNN:AUC = %0.4f' % roc_auc_knn)
    # plt.plot(false_positive_rate_cblof, true_positive_rate_cblof, label='CBLOF:AUC = %0.4f' % roc_auc_cblof)
    plt.plot(false_positive_rate_lof, true_positive_rate_lof, label='LOF:AUC = %0.4f' % roc_auc_lof)
    plt.plot(false_positive_rate_if, true_positive_rate_if, label='Isolation Forest:AUC = %0.4f' % roc_auc_if)
    plt.plot(false_positive_rate_fg, true_positive_rate_fg, label='Feature Bagging:AUC = %0.4f' % roc_auc_fg)
    plt.plot(false_positive_rate_hbos, true_positive_rate_hbos, label='HBOS:AUC = %0.4f' % roc_auc_hbos)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

if __name__=='__main__':
    path = 'data/wine_benchmarks/wine/benchmarks/'
    filelist=os.listdir(path)
    for file in filelist:
        detection(path+file,file)
    df=DataFrame(dict)
    df.to_csv('result1.csv')
    print("\nOn Training Data:")
    print('Angle-based Outlier Detector (ABOD) ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_abod_train']),prn_avg=mean(dict['prn_abod_train'])))
    print('K Nearest Neighbors (KNN) ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_knn_train']), prn_avg=mean(dict['prn_knn_train'])))
    print('Feature Bagging ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_fg_train']), prn_avg=mean(dict['prn_fg_train'])))
    print('Histogram-base Outlier Detection (HBOS) ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_hbos_train']), prn_avg=mean(dict['prn_hbos_train'])))
    print('Isolation Forest ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_if_train']), prn_avg=mean(dict['prn_if_train'])))
    print('LOF ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_lof_train']), prn_avg=mean(dict['prn_lof_train'])))
    print("\nOn Test Data:")
    print('Angle-based Outlier Detector (ABOD) ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_abod_test']), prn_avg=mean(dict['prn_abod_test'])))
    print('K Nearest Neighbors (KNN) ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_knn_test']), prn_avg=mean(dict['prn_knn_test'])))
    print('Feature Bagging ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_fg_test']), prn_avg=mean(dict['prn_fg_test'])))
    print('Histogram-base Outlier Detection (HBOS) ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_hbos_test']), prn_avg=mean(dict['prn_hbos_test'])))
    print('Isolation Forest ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_if_test']), prn_avg=mean(dict['prn_if_test'])))
    print('LOF ROC(average):{roc_avg}, precision @ rank n(average):{prn_avg}'.format(
        roc_avg=mean(dict['roc_lof_test']), prn_avg=mean(dict['prn_lof_test'])))
    plot_roc(draw)
