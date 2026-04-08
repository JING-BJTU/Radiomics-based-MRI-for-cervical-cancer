#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from scipy import stats
from itertools import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,f_classif,chi2,SelectPercentile
from sklearn.linear_model import lasso_path,LassoCV
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import sem
from pandas import Series,DataFrame
from sklearn.model_selection import StratifiedKFold
# from scipy import interp
from joblib import dump, load
from sklearn.feature_selection import SelectFromModel
import pickle
import os
import scipy
from sklearn.preprocessing import label_binarize
matplotlib.use('Agg')
dpi_set = 300



from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import BorderlineSMOTE, SVMSMOTE

def smote_data(features, y, random, data_path):
    
    # sm = SVMSMOTE(random_state=random, sampling_strategy={0:220,1:220})
    # sm = ADASYN(random_state=random, sampling_strategy={0:150,1:150})
    # sm = BorderlineSMOTE(random_state=random, sampling_strategy={0: 150, 1: 150})
    sm = BorderlineSMOTE(random_state=random)

    print('lenth of feature = ', len(features))
    print('lenth of y = ', len(y))
    
    features_plus, y_plus = sm.fit_resample(features, pd.DataFrame(y))
    print(" {} data with {} features each.".format(*features_plus.shape))
    print('lenth of plus y = ', len(y_plus))

    features_plus=pd.DataFrame(features_plus, columns=features.columns)
    #特定行取整
    
    y_plus=pd.DataFrame(y_plus)

    data_plus = pd.concat([y_plus, features_plus], axis=1, join='inner')
    data_plus.to_csv(os.path.join(data_path, 'data_plus.csv'))

    return features_plus, y_plus


def move_lowvariance(X, y, out_path):

    fs_path = os.path.join(out_path, 'fs')
    if not os.path.isdir(fs_path):
        os.makedirs(fs_path)

    sel = VarianceThreshold(threshold=(.08 * (1 - .08)))
    sel_X = sel.fit_transform(X)
    result = sel.get_support()
    features = X.columns
    features_split = []
    sel_features = []
    sel_features_split = []
    first = []
    sel_first = []
    for index, item in enumerate(features):
        item_split = item.split('_')
        features_split.append(item_split)
        first.append(item_split[0])
        if result[index]:
            sel_features.append(item)
            sel_features_split.append(item_split)
            sel_first.append(item_split[0])
    print("features reduced from {0} to {1} by move_lowvariance".format(len(features),len(sel_features)))
    X_sel = X[sel_features]
    y = y
    features = sel_features
    X_sel.to_csv(os.path.join(fs_path, 'lowVariance.csv'))
    return X_sel, y, features


def select_KBest(X,y,out_path):
    
    fs_path = os.path.join(out_path, 'fs')
    if not os.path.isdir(fs_path):
        os.makedirs(fs_path)
    
    sel = SelectKBest(f_classif, k='all').fit(X, np.ravel(y))
    scores = sel.scores_
    pvalue = sel.pvalues_
    features = sel.feature_names_in_
    result = pd.DataFrame([scores, pvalue], columns=features).T
    result.columns = ['score', 'p']
    sel_features_p = result[result['p'] < 0.05]
    sel_features_p = sel_features_p.sort_values(by='p', ascending=False)

    # if len(features) >1000:
    #     sel_features_p = sel_features_p.iloc[:40,:]
    # else:
    #     # sel_features_p = sel_features_p[sel_features_p['p'] > 0.01]
    #     sel_features_p = sel_features_p.iloc[:50,:]

    sel_features = sel_features_p.index
    X = X[sel_features]
    y =y
    print("features reduced from {0} to {1} by selectkbest".format(len(features),len(sel_features)))

    # print (stats_result)

    result.to_csv(os.path.join(fs_path, 'all_P_score.csv'))
    sel_features_p.to_csv(os.path.join(fs_path, 'sk_P_score.csv'))

    X.to_csv(os.path.join(fs_path, 'selectKbest.csv'))
    
    return X, y, sel_features


def lasso_filter(X, y, cv, max_iter, out_path):
    
    fs_path = os.path.join(out_path, 'fs')
    if not os.path.isdir(fs_path):
        os.makedirs(fs_path)
    
    model = LassoCV(cv=cv, max_iter=max_iter, random_state=42).fit(X, y)  # n_jobs=-1, tol=0.0001
    selected = -np.log10(model.alpha_)
    m_log_alphas = -np.log10(model.alphas_)
    features = X.columns
    coef = pd.Series(model.coef_, index=features)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label='alpha: CV estimate')
    plt.legend()
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.savefig(os.path.join(fs_path, "lasso1.png"), dpi=dpi_set)
    # plt.show()
    plt.close()

    alphas_lasso, coefs_lasso, _ = lasso_path(X, np.ravel(y))
    plt.figure()
    ax = plt.gca()
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',label='alpha: CV estimate')
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso Path')
    plt.axis('tight')
    plt.savefig(os.path.join(fs_path, "lasso2.png"), dpi=dpi_set)
    # plt.show()
    plt.close()

    # selected_coef = coef[coef != 0]
    selected_coef = coef[np.abs(coef) > 0.001]
    lasso_feature = selected_coef.index
    lasso_result = X[lasso_feature]
    lasso_result.to_csv(os.path.join(fs_path, 'lasso_sel.csv')) #需要可导出
    print("features reduced from {0} to {1} by lasso".format(len(features), len(selected_coef)))

    y_Xsel = pd.concat([pd.DataFrame(y), lasso_result], axis=1, join='inner', ignore_index=False)
    y_Xsel.to_csv(os.path.join(fs_path, 'y_Xsel.csv'))
    
    lasso_result.to_csv(os.path.join(fs_path, 'x_train_sel.csv')) #需要可导出
    
    mp_coef = pd.concat([selected_coef.sort_values()])
    print(len(mp_coef))
    # print mp_coef
    mp_coef.to_csv(os.path.join(fs_path, 'lasso-xishu.csv'))
    #
    if len(lasso_result.columns) > 3:
        lasso_result = VIF_sel(lasso_result, out_path)
        lasso_result.to_csv(os.path.join(out_path, 'x_train_sel.csv'), encoding='936')
        print(lasso_result.shape)

    return lasso_result, y, selected_coef, mp_coef


from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(df):
    vif = pd.DataFrame()
    vif['index'] = df.columns
    vif['VIF'] = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
    return vif

def VIF_sel(feature, out_path):
    # https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
    # https://zhuanlan.zhihu.com/p/435430759
    # VIF dataframe
    # calculating VIF for each feature

    feature_num = len(feature.columns)
    vif = calculate_vif(feature)
    while (vif['VIF'] > 10).any():
        remove = vif.sort_values(by='VIF', ascending=False)['index'][:1].values[0]
        feature.drop(remove, axis=1, inplace=True)
        vif = calculate_vif(feature)

    # feature_sel_col = vif.iloc[:-1, :]['index'].values
    feature_sel_col = pd.DataFrame(vif)
    feature_sel_col.to_csv(os.path.join(out_path, 'vif_data.csv'), encoding='936')

    feature_sel = feature[vif['index']]

    print("features reduced from {0} to {1} by viff".format(feature_num, len(feature_sel.columns)))

    return feature_sel


def auc_report(y_test, y_pred_prob, clf_path):
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    auc_result = roc_auc_score(y_test, y_pred_prob)
    #figure
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(fpr, tpr, label='AUC is %0.3f'%auc_result)
    plt.rcParams['font.size'] = 8
    plt.legend(loc="lower right",fontsize=8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    # plt.title('ROC curve for T2 classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.legend(fontsize=8)
    plt.savefig(os.path.join(clf_path, "auc.png"), dpi=dpi_set)
    # plt.show()
    plt.close()

    # %95CI
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred_prob) - 1, len(y_pred_prob))
        if len(np.unique(y_test.iloc[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_test.iloc[indices], y_pred_prob.iloc[indices])
        bootstrapped_scores.append(score)
    #         print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    # print("The Data Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))

    # cut-off point
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]
    threshold_value = roc_t['threshold'].values
    # print(threshold_value)
    # print('auc = ', auc_result)

    # 以threshold_value值为阈值，大于的为1，＜的为0，生成预测值y_pred_threshold,并返回，用来生成混淆矩阵
    y_pred_threshold = []
    for num in y_pred_prob.values:
        if num > threshold_value:
            y_pred_threshold.append(1)
        else:
            y_pred_threshold.append(0)
    y_pred_threshold = DataFrame(y_pred_threshold, index=y_test.index, columns=['y_pred'])

    return y_test, y_pred_threshold, auc_result, threshold_value, confidence_lower, confidence_upper


from sklearn import metrics
def plot_matrix(y_true, y_pred, labels_name, save_path, title=None, thresh=0.8, axis_labels=None):
# 利用sklearn中的函数生成混淆矩阵并归一化
    cm = metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true), sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

# 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.colorbar()  # 绘制图例

# 图像标题
    if title is not None:
        plt.title(title)
# 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# 显示
#     pl.show()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi_set)
    # plt.show()
    plt.clf()
    plt.close()


def y_pred_cal(label, y_prob, threshold_value_recal, sk_threshold_value):

    y_prob_ori = np.array(y_prob, dtype=float)
    y_prob = (y_prob_ori - y_prob_ori.min()) / (y_prob_ori.max() - y_prob_ori.min())

    if threshold_value_recal:
        threshold_value = sk_threshold_value
    else:
        threshold_value,  point = Find_yudeng_Cutoff(label, y_prob)
    y_pred_threshold = []
    for num in y_prob[:, 0]:
        if num >= threshold_value:
            y_pred_threshold.append(1)
        else:
            y_pred_threshold.append(0)
    y_pred_threshold = pd.DataFrame(y_pred_threshold, index=label.index, columns=['y_pred'])

    return threshold_value, y_pred_threshold

def Find_yudeng_Cutoff(label, y_prob):
    FPR, TPR, thresholds = roc_curve(label, y_prob, pos_label=1)
    y = TPR - FPR
    Youden_index = np.argmax(y) # Only the first occurrence is returned.
    optimal_threshold = thresholds[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]

    return optimal_threshold, point


# 混淆矩阵计算
def model_training_CV(X_radiomics, y, clf, clf_path):
    cv = 5
    y = np.array(y, dtype=int)

    X_index = X_radiomics.index
    # X_index = DataFrame(X_index, columns=['X_index'])
    X = np.array(X_radiomics)

    yprob = DataFrame()

    cv = StratifiedKFold(n_splits=cv)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 1
    auc_best = 0.5
    y_pred_threshold = []
    for train, test in cv.split(X, y):
        clf = clf.fit(X[train], y[train])
        probas_ = clf.predict_proba(X[test])
        pred_ = clf.predict(X[test])
        # print probas_
        # print pred_
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        plt.rcParams['font.size'] = 8
        # plt.legend(fontsize=8)
        i += 1
        if roc_auc > auc_best:
            auc_best = roc_auc
            print('auc_best = ', auc_best)
            probas_ = clf.fit(X[train], y[train]).predict_proba(X[train])
            pred_ = clf.fit(X[train], y[train]).predict(X[train])
            model = pickle.dumps(clf)
            y_label_train = y[train]
            y_pred_probas = probas_[:, 1]
            y_pred_threshold = pred_
            y_true = DataFrame(y_label_train, columns=['y_true'])
            y_pred_probas = DataFrame(y_pred_probas, columns=['y_proba0'])
            y_pred_threshold = DataFrame(y_pred_threshold, columns=['y_pred'])
            y_true_pred = pd.concat([y_true, y_pred_threshold, y_pred_probas], axis=1, join='inner', ignore_index=False)
            y_true_pred.to_csv(os.path.join(clf_path, 'yTrain_true_pred.csv'))

            X_train = DataFrame(X[train], columns=X_radiomics.columns)
            y_train_X = pd.concat([y_true_pred, X_train], axis=1, join='inner',
                                  ignore_index=False)
            y_train_X.to_csv(os.path.join(clf_path, 'yTrain_true_pred_X.csv'))
            # 测试集
            probas_test = clf.predict_proba(X[test])
            pred_test = clf.predict(X[test])
            y_label_test = y[test]
            y_pred_probas_test = probas_test[:, 1]
            y_pred_threshold_test = pred_test
            y_true_test = DataFrame(y_label_test, columns=['y_true'])
            y_pred_probas_test = DataFrame(y_pred_probas_test, columns=['y_proba0'])
            y_pred_threshold_test = DataFrame(y_pred_threshold_test, columns=['y_pred'])
            y_true_pred_test = pd.concat([y_true_test, y_pred_threshold_test, y_pred_probas_test], axis=1, join='inner',
                                         ignore_index=False)
            y_true_pred_test.to_csv(os.path.join(clf_path, 'yVal_true_pred.csv'))

            X_test = DataFrame(X[test], columns=X_radiomics.columns)
            y_test_X = pd.concat([y_true_pred_test, X_test], axis=1, join='inner',
                                 ignore_index=False)
            y_test_X.to_csv(os.path.join(clf_path, 'yVal_true_pred_X.csv'))

            model = pickle.dumps(clf)
            dump(clf, os.path.join(clf_path, 'model.joblib'))

        '''
        yprobabi = DataFrame(probas_,index = test, columns = ['y_proba0','y_proba1'])
        y_true = DataFrame(y[test],index = test,columns = ['y_true'])
        y_pred = DataFrame(pred_,index = test,columns = ['y_pred'])
        yprobabi = pd.concat([y_true,y_pred,yprobabi],axis=1,join='inner',ignore_index = True)
        yprob = yprob.append(yprobabi)
        y_proba = yprob.sort_index()
        '''

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]
    threshold_value = roc_t['threshold'].values
    print('threshold_value = {}'.format(threshold_value))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print('mean_auc = {}'.format(mean_auc))

    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(os.path.join(clf_path, "train_auc.png"), dpi=dpi_set)
    # plt.show()
    plt.close()

    if auc_best == 0.5:
        probas_ = clf.fit(X, y).predict_proba(X)
        pred_ = clf.fit(X, y).predict(X)
        model = pickle.dumps(clf)

        mean_auc = 0.5
        threshold_value = 0.5
        ############以threshold_value值为阈值，大于的为1，＜的为0，生成预测值y_pred_threshold,并返回，用来生成混淆矩阵
        y_pred_probas = probas_[:, 1]

        y_pred_threshold = []
        for num in y_pred_probas:
            if num > threshold_value:
                y_pred_threshold.append(1)
            else:
                y_pred_threshold.append(0)
        y_true_test = DataFrame(y, index=X_index, columns=['y_true'])
        y_pred_probas_test = DataFrame(y_pred_probas, index=X_index, columns=['y_proba0'])
        y_pred_threshold_test = DataFrame(y_pred_threshold, index=X_index, columns=['y_pred'])

        y_true_pred = pd.concat([y_true_test, y_pred_threshold_test, y_pred_probas_test], axis=1, join='inner',
                                ignore_index=False)
        y_true_pred.to_csv(os.path.join(clf_path, 'yTrain_all_true_pred.csv'))
        y_label_train = y
        y_train_X = X


    return model, mean_auc, y_true_test, y_pred_threshold_test, y_pred_probas_test, threshold_value, y_train_X


def model_training_CV_all(X_radiomics, y, clf, random, clf_path):
    cv = 5
    y = np.array(y, dtype=int)

    X_index = X_radiomics.index
    # X_index = DataFrame(X_index, columns=['X_index'])
    X = np.array(X_radiomics)


    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    data_performance = []
    fold_i_train = 1
    for train, test in cv.split(X, y):
        clf = clf.fit(X[train], y[train])
        probas_ = clf.predict_proba(X[train])
        pred_ = clf.fit(X[train], y[train]).predict(X[train])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[train], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (fold_i_train, roc_auc))
        plt.rcParams['font.size'] = 8
        # plt.legend(fontsize=8)

        y_true = DataFrame(y[train], columns=['y_true'])
        y_pred_probas = DataFrame(probas_[:,1], columns=['y_proba0'])
        y_pred_threshold = DataFrame(pred_, columns=['y_pred'])
        y_true_pred = pd.concat([y_true, y_pred_threshold, y_pred_probas], axis=1, join='inner', ignore_index=False)
        y_true_pred.to_csv(os.path.join(clf_path, 'yTrain_true_pred_{}.csv'.format(str(fold_i_train))))

        X_train = DataFrame(X[train], columns=X_radiomics.columns)
        y_train_X = pd.concat([y_true_pred, X_train], axis=1, join='inner',
                             ignore_index=False)
        y_train_X.to_csv(os.path.join(clf_path, 'yTrain_true_pred_X_{}.csv'.format(str(fold_i_train))))

        out_path_train_clf = os.path.join(clf_path, clf.__class__.__name__+'_train_'+str(fold_i_train))
        tra_acc, tra_sen, tra_spe, tra_CI_L, tra_CI_H = analysis_report(y[train], y_pred_probas,
                                                                        out_path_train_clf)

        result_all = ['train', fold_i_train, roc_auc, tra_acc, tra_sen, tra_spe, tra_CI_L, tra_CI_H]
        data_performance.append(result_all)

        fold_i_train = fold_i_train + 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]
    threshold_value = roc_t['threshold'].values
    print('threshold_value = {}'.format(threshold_value))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print(mean_auc)

    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(os.path.join(clf_path, "train_auc.png"), dpi=dpi_set)
    # plt.show()
    plt.close()

    t = [data_performance[i][1:] for i in range(len(data_performance) - 5, len(data_performance))]
    data_mean = np.average(t, axis=0)
    data_performance_mean = np.r_[['mean'],data_mean]
    data_performance.append(list(data_performance_mean))

    fold_i_test = 1
    tprs_test = []
    aucs_test = []
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        y_pred_probas_test = probas_[:, 1]
        pred_ = clf.fit(X[train], y[train]).predict(X[test])
        # print probas_
        # print pred_
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs_test.append(np.interp(mean_fpr, fpr, tpr))
        tprs_test[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs_test.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (fold_i_test, roc_auc))
        plt.rcParams['font.size'] = 8
        # plt.legend(fontsize=8)

        y_true_test = DataFrame(y[test], columns=['y_true'])
        y_pred_probas_test = DataFrame(y_pred_probas_test, columns=['y_proba0'])
        y_pred_threshold_test = DataFrame(pred_, columns=['y_pred'])
        y_true_pred_test = pd.concat([y_true_test, y_pred_threshold_test, y_pred_probas_test], axis=1, join='inner',
                                     ignore_index=False)
        y_true_pred_test.to_csv(os.path.join(clf_path, 'yVal_true_pred_{}.csv'.format(str(fold_i_test))))

        X_test = DataFrame(X[test], columns=X_radiomics.columns)
        y_test_X = pd.concat([y_true_pred_test, X_test], axis=1, join='inner',
                                     ignore_index=False)
        y_test_X.to_csv(os.path.join(clf_path, 'yVal_true_pred_X_{}.csv'.format(str(fold_i_test))))

        out_path_test_clf = os.path.join(clf_path, clf.__class__.__name__ + '_train_' + str(fold_i_test))
        test_acc, test_sen, test_spe, test_CI_L, test_CI_H = analysis_report(y[test], y_pred_probas_test,
                                                                             out_path_test_clf)
        result_all = ['test', fold_i_test,  roc_auc, test_acc, test_sen, test_spe, test_CI_L, test_CI_H]
        data_performance.append(result_all)
        fold_i_test = fold_i_test + 1


    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    j = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=j), 'threshold': pd.Series(thresholds, index=j)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]
    threshold_value = roc_t['threshold'].values
    print('threshold_value = {}'.format(threshold_value))

    mean_tpr = np.mean(tprs_test, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc_test = auc(mean_fpr, mean_tpr)
    print(mean_auc_test)

    std_auc = np.std(aucs_test)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc_test, std_auc), lw=2,
             alpha=.8)

    std_tpr = np.std(tprs_test, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(os.path.join(clf_path, "test_auc.png"), dpi=dpi_set)
    # plt.show()
    plt.close()

    t = [data_performance[i][1:] for i in range(len(data_performance) - 5, len(data_performance))]
    data_mean = np.average(t, axis=0)
    data_performance_mean = np.r_[['mean'],data_mean]
    data_performance.append(list(data_performance_mean))
    
    return data_performance


def model_training(X, y, clf, clf_path):

    probas_ = clf.fit(X, y).predict_proba(X)
    pred_ = clf.fit(X, y).predict(X)
    probas_ = DataFrame(probas_[:, 1], index=X.index)
    pred_ = DataFrame(pred_, index=X.index)
    y_df = pd.DataFrame(y, index=X.index)

    y, y_pred, auc_result, threshold_value, confidence_lower, confidence_upper = auc_report(y_df, probas_, clf_path)
    print("model_training: auc = ", auc_result)

    y_pred_threshold = []
    for num in probas_.values:
        if num > threshold_value:
            y_pred_threshold.append(1)
        else:
            y_pred_threshold.append(0)

    model = pickle.dumps(clf)

    y_pred_threshold = DataFrame(y_pred_threshold, columns=['y_pred'], index=y.index)
    y_true_pred = pd.concat([y_df, probas_, y_pred_threshold], axis=1, join='inner', ignore_index=False)
    y_true_pred.to_csv(os.path.join(clf_path, 'yTrain_true_pred.csv'))

    y_train_X = pd.concat([y_true_pred, X], axis=1, join='inner',
                          ignore_index=False)
    y_train_X.to_csv(os.path.join(clf_path, 'yTrain_true_pred_X.csv'))


    return model, auc_result, y_df, y_pred_threshold, probas_, threshold_value, y_train_X


def model_testing(model, X_test, y_test, clf_path):

    X_test.to_csv(os.path.join(clf_path,'x_test_sel.csv'))
    
    X_index = X_test.index

    clf = pickle.loads(model)
    y_pred_ori = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    y_pred_prob = DataFrame(y_pred_prob, index=X_index, columns=['y_proba0'])
    y_pred = DataFrame(y_pred_ori, index=X_index, columns=['y_pred'])
    
    yTest_true_pred = pd.concat([y_test, y_pred, y_pred_prob], axis=1, join='inner', ignore_index=False)
    yTest_true_pred.to_csv(os.path.join(clf_path, 'yTest_true_pred.csv'))

    yTest_true_pred_X = pd.concat([y_test, y_pred, y_pred_prob, X_test], axis=1, join='inner', ignore_index=False)
    yTest_true_pred_X.to_csv(os.path.join(clf_path, 'yTest_true_pred_X.csv'))

    return y_pred, y_pred_prob


def rad_score(feature, coef, modal, path_in, Train_test_tag):
    feature_name = feature.columns
    coefficient = coef[feature_name]

    radscore = np.dot(feature, coefficient)
    radscore = pd.DataFrame(radscore, index=feature.index)
    radscore.columns = ['rad_score_{}'.format(modal)]

    radscore.to_csv(os.path.join(path_in, 'radscore_{}_{}.csv'.format(modal, Train_test_tag)))
    return radscore


def spe_sen_acc_pre_f1(Y_test, Y_pred, n):
    """
    https://blog.csdn.net/qq_44786208/article/details/115672926
    Args:
        Y_test:
        Y_pred:
        n:

    Returns:

    """
    spe = []
    sen = []
    pre = []
    acc_tt = 0
    con_mat = confusion_matrix(Y_test, Y_pred)
    number = np.sum(con_mat[:, :])

    if n>2:
        for i in range(n):
            tp = con_mat[i][i]
            fn = np.sum(con_mat[i, :]) - tp
            fp = np.sum(con_mat[:, i]) - tp
            tn = number - tp - fn - fp
            spe1 = tn / (tn + fp)
            sen1 = tp / (tp + fn)
            pre1 = tp / (tp + fp)
            spe.append(spe1)
            sen.append(sen1)
            pre.append(pre1)
            acc_tt = acc_tt+tp

    else:

        tp = con_mat[1][1]
        fn = con_mat[1][0]
        fp = con_mat[0][1]
        tn = con_mat[0][0]
        spe1 = tn / (tn + fp)
        sen1 = tp / (tp + fn)
        pre1 = tp / (tp + fp)
        spe.append(spe1)
        sen.append(sen1)
        pre.append(pre1)
        acc_tt = tp+tn

    return np.mean(sen), np.mean(spe), np.mean(pre), (np.mean(sen)+np.mean(pre))/2, acc_tt/number


def analysis_report(label, y_pred, save_path, group_tag, curve_name):
    # labels = list(set(y_true))
    # y_true = np.array(label, dtype=int)

    # png_path = os.path.join(save_path, "CM_{}_{}.tiff".format(group_tag, curve_name))
    # plot_matrix(label, y_pred, np.unique(label), save_path = png_path)

    result0 = classification_report(label, y_pred, output_dict=True)

    df_result = pd.DataFrame(result0).transpose()
    # csv_save = os.path.join(save_path, 'result_{}_{}.csv'.format(group_tag, curve_name))
    # df_result.to_csv(csv_save)

    Sensitivity, Specificity, Precision, f1_score, Accuracy = spe_sen_acc_pre_f1(label, y_pred, len(np.unique(label)))

    # Accuracy = df_result.iloc[-3, 0]
    # Sensitivity = sen(label, y_pred, len(np.unique(label)))
    # Specificity = spe(label, y_pred, len(np.unique(label)))
    # Precision = df_result.iloc[-2,0]
    # f1_score = df_result.iloc[-2,2]

    return Accuracy, Sensitivity, Specificity, Precision, f1_score
