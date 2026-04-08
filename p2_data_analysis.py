#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.linear_model import LogisticRegression

import sys
from fs_ml import train_test, select_KBest, lasso_filter, move_lowvariance, smote_data
from fs_ml import model_training_CV, model_training, model_testing, \
    multi_model_train, multi_class_results_show, mul_model_testing
from fs_ml import auc_report, analysis_report, rad_score
import time
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('ignore')


def result_logger(input_path, s):
    with open(os.path.join(input_path, 'log.txt'), 'a+') as f:
        f.write(s)


def split(full_df, ran_dom, ratio):

    # full_df = full_df.sort_index()
    full_list = shuffle(full_df, random_state=ran_dom)

    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    grouped = full_list.groupby(full_list.iloc[:, 0])
    for k, g in grouped:
        uniquekeys = k
        # print(uniquekeys)
        y_i = grouped.get_group(uniquekeys)

        offset = int(len(y_i) * ratio)

        y_i_list = shuffle(y_i, random_state=ran_dom)
        y_tr = y_i_list.iloc[offset:]
        y_te = y_i_list.iloc[:offset]
        y_train = pd.concat([y_train, y_tr], axis=0)
        y_test = pd.concat([y_test, y_te], axis=0)

    return y_train, y_test


def data_std(train_data, test_data, tag):
    if tag == 'all':
        all_data = pd.concat([train_data, test_data], join='inner')
        ss = StandardScaler().fit(all_data)
        all_data_std = ss.transform(all_data)
        all_data_std = pd.DataFrame(all_data_std, index=all_data.index, columns=all_data.columns)
        train_data_std = pd.DataFrame(all_data_std, index = train_data.index)
        test_data_std = pd.DataFrame(all_data_std, index = test_data.index)

    if tag == 'train':

        ss = StandardScaler().fit(train_data)
        train_data_std = ss.transform(train_data)
        train_data_std = pd.DataFrame(train_data_std, index=train_data.index, columns=train_data.columns)

        test_data_std = ss.transform(test_data)
        test_data_std = pd.DataFrame(test_data_std, index = test_data.index, columns=train_data.columns)

    if tag == 'single':

        ss = StandardScaler().fit(train_data)
        train_data_std = ss.transform(train_data)
        train_data_std = pd.DataFrame(train_data_std, index=train_data.index, columns=train_data.columns)

        ss1 = StandardScaler().fit(test_data)
        test_data_std = ss1.transform(test_data)
        test_data_std = pd.DataFrame(test_data_std, index = test_data.index, columns=test_data.columns)

    return train_data_std, test_data_std


def feature_sel(X_train, X_test, label_train, tag, out_path, return_dict=True):
    # --------------计算训练数据的均值和方差

    ss = StandardScaler().fit(X_train)
    train_data_std = ss.transform(X_train)
    train_data_std = pd.DataFrame(train_data_std, index=X_train.index, columns=X_train.columns)

    test_data_std = ss.transform(X_test)
    X_test_all_std = pd.DataFrame(test_data_std, index = X_test.index, columns=X_train.columns)


    # ---------特征选择

    if train_data_std.shape[1] < 4:
        X_train_sel = train_data_std.copy()
        X_test_sel = X_test_all_std.copy()

        model_lr = LogisticRegression().fit(X_train_sel, label_train)
        intercept_ = pd.DataFrame(model_lr.intercept_, columns=['intercept'])
        intercept_.to_csv(os.path.join(out_path, 'clf_LR_train_intercept.csv'))
        coef_ = model_lr.coef_
        feature_names_in_ = model_lr.feature_names_in_
        clf_LR_coef = pd.DataFrame(coef_.T, index=feature_names_in_)
        clf_LR_coef.to_csv(os.path.join(out_path, 'clf_LR_train_coef.csv'))
        radsTrain = pd.DataFrame(model_lr.predict_proba(X_train_sel)[:, 1], index = X_train_sel.index)
        radsTrain.columns = ['RS_{}'.format(tag)]
        radsTest = pd.DataFrame(model_lr.predict_proba(X_test_sel)[:, 1], index = X_test_sel.index)
        radsTest.columns = ['RS_{}'.format(tag)]

    else:
        X_train_sel, y, features_name = move_lowvariance(train_data_std, label_train, out_path)
        X_train_sel, y, features_name = select_KBest(X_train_sel, label_train, out_path)

        try:

            max_iter = 500
            X_train_sel, y, features_name, coef = lasso_filter(X_train_sel, y, 5, max_iter, out_path)
            radsTrain = rad_score(X_train_sel, coef, tag, out_path, 'train')

            X_test_sel = pd.DataFrame(X_test_all_std, columns=X_train_sel.columns)
            radsTest = rad_score(X_test_sel, coef, tag, out_path, 'test')

        except:
            print('lasso ValueError: at least one array or dtype is required')
            radsTrain=[]
            radsTest=[]
            X_test_sel = pd.DataFrame(X_test_all_std, columns=X_train_sel.columns)

    if return_dict:
        return  {'X_train_sel': X_train_sel,
                 'X_test_sel': X_test_sel,
                 'radsTrain': radsTrain,
                 'radsTest': radsTest
                 }
    else:
        return X_train_sel, X_test_sel, radsTrain, radsTest



def data_all(input_path):

    label_data = pd.read_csv(os.path.join(input_path, 'Label.csv'), index_col=0)

    data = pd.read_csv(os.path.join(input_path, 'DATA_ICC_75_Inter.csv'), index_col=0)

    data_1 = list(filter(lambda x: len(x) != len(x.replace('T1C', '')), list(data.columns)))
    data_2 = list(filter(lambda x: len(x) != len(x.replace('T2', '')), list(data.columns)))
    data_3 = list(filter(lambda x: len(x) != len(x.replace('DWI', '')), list(data.columns)))

    feature_4 = pd.concat([data[data_1], data[data_2]], axis=1, join='inner')
    feature_5 = data.copy()

    label_dict = {
                  'Label': label_data,
                  }
    # 'dti': dti_feature
    feature_dict = {
                    'T1C': data[data_1],
                    'T2': data[data_2],
                    'DWI': data[data_3],
                    'JointModel1': feature_4,
                    'JointModel2':feature_5
    }

    return label_dict, feature_dict


def analysis_data():

    input_path = './input_path'
    out_path_level1 = './result'
    if not os.path.exists(out_path_level1):
        os.system('mkdir ' + out_path_level1)

    # 循环所有分类的所有模态
    result = []

    label_dict, feature_dict = data_all(input_path)

    model_lr = LogisticRegression(penalty="l2", class_weight='balanced')

    model_list = [model_lr]
    model_name = ['ML_LR']


    # 固定训练集与测试集
    ratio_2 = 0.2
    ran_statue = 130

    print('random = {}'.format(ran_statue))
    out_path_level2 = os.path.join(out_path_level1, 'exp_' + str(ran_statue))
    if not os.path.exists(out_path_level2):
        os.system('mkdir ' + out_path_level2)

    for label_name, label_group_i in label_dict.items():
        out_path_level3 = os.path.join(out_path_level2, str(label_name))
        if not os.path.exists(out_path_level3):
            os.system('mkdir ' + out_path_level3)

        train_label, test_label = split(label_group_i, ratio=ratio_2, ran_dom=ran_statue)
        feature_radscore_list = []

        for moda_name_i, exp_radiomics in feature_dict.items():
            print('mod_name_i = {}'.format(moda_name_i))
            out_path = os.path.join(out_path_level3, moda_name_i)
            if not os.path.exists(out_path):
                os.system('mkdir ' + out_path)

            y_radiomics_train = pd.merge(train_label, exp_radiomics, how='left', on='ResearchId')
            train_data = y_radiomics_train.iloc[:, 1:]
            train_data = train_data.fillna(train_data.mean())
            train_label = y_radiomics_train.iloc[:, 0]

            y_radiomics_test = pd.merge(test_label, exp_radiomics, how='left', on='ResearchId')
            test_data = y_radiomics_test.iloc[:, 1:]
            test_data = test_data.fillna(test_data.mean())
            test_label = y_radiomics_test.iloc[:, 0]
            test_label = pd.DataFrame(test_label)

            feature_radscore_dict = feature_sel(train_data, test_data, train_label, moda_name_i,
                                                out_path, return_dict=True)

            feature_radscore_list.append(feature_radscore_dict)

        # -----------模型构建

        try:

            M1_train = feature_radscore_list[0]['X_train_sel']
            M1_test = feature_radscore_list[0]['X_test_sel']
            M1_coef = feature_radscore_list[0]['lasso_coef']

            M2_train = feature_radscore_list[1]['X_train_sel']
            M2_test = feature_radscore_list[1]['X_test_sel']
            M2_coef = feature_radscore_list[1]['lasso_coef']

            M3_train = feature_radscore_list[2]['X_train_sel']
            M3_test = feature_radscore_list[2]['X_test_sel']
            M3_coef = feature_radscore_list[2]['lasso_coef']

            M4_train = feature_radscore_list[3]['X_train_sel']
            M4_test = feature_radscore_list[3]['X_test_sel']
            M4_coef = feature_radscore_list[3]['lasso_coef']
            
            M5_train = feature_radscore_list[4]['X_train_sel']
            M5_test = feature_radscore_list[4]['X_test_sel']
            M5_coef = feature_radscore_list[4]['lasso_coef']

        except:
            continue

        X_train_sets = [M1_train, M2_train, M3_train, M4_train, M5_train]
        X_test_sets = [M1_test, M2_test, M3_test, M4_test, M5_test]
        coef_set = [M1_coef, M2_coef, M3_coef, M4_coef, M5_coef]
        model_name_list = ['T1C', 'T2', 'DWI', 'JointModel1', 'JointModel2']

        for model_index, moda_name_i in enumerate(model_name_list):
            X_train_sel = X_train_sets[model_index]
            X_test_sel_std = X_test_sets[model_index]
            coef_lasso = coef_set[model_index]


            for index, clf in enumerate(model_list):

                model_name_i = model_name[index]
                # out_path_clf = os.path.join(out_path_level3, moda_name_i+"_"+model_name_i)
                print(ran_statue, model_name_i, moda_name_i)

                out_path_clf = out_path_level3+'/'+ moda_name_i + "/" + model_name_i
                if not os.path.exists(out_path_clf):
                    os.system('mkdir ' + out_path_clf)
                # clf = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')
                # clf = svm(class_weight='balanced')

                out_path_train_clf = os.path.join(out_path_clf, 'train')
                out_path_test_clf = os.path.join(out_path_clf, 'test')

                if not os.path.exists(out_path_train_clf):
                    os.system('mkdir ' + out_path_train_clf)
                    os.system('mkdir ' + out_path_test_clf)


                if len(X_train_sel.columns)<2:
                    continue

                # model, mean_auc, y_label_train, y_pred_threshold, y_probas, threshold_value, y_train_X = model_training_CV(X_train_sel, np.ravel(train_label), clf, out_path_train_clf)
                model, mean_auc, y_label_train, y_pred_threshold, y_probas, threshold_value, y_train_X = model_training(X_train_sel, np.ravel(train_label), clf, out_path_train_clf)
                y_label_train, y_pred_threshold, auc_result_train, threshold_value, tra_CI_L, tra_CI_H = auc_report(y_label_train, y_probas, out_path_train_clf)
                print("{} model_training or validation: auc = {}".format(moda_name_i, auc_result_train))
                result_logger(out_path_train_clf, "model_training or validation: auc = {}\n".format(auc_result_train))
                tra_acc, tra_sen, tra_spe, tra_precision, tra_f1 = analysis_report(y_label_train, y_pred_threshold,
                                                                                out_path_train_clf, label_group_i,moda_name_i+"_"+model_name_i)
                # print(tra_acc, tra_sen, tra_spe, tra_CI_L, tra_CI_H)

                # ----------测试集测试
                y_Xsel_test1 = pd.concat([test_label, X_test_sel_std], axis=1)
                y_Xsel_test1.to_csv(os.path.join(out_path_level3, moda_name_i + '/yx_test1_sel.csv'))

                y_pred_test, y_proba_test = model_testing(model, X_test_sel_std, test_label, out_path_test_clf)
                y_test, y_pred_threshold, auc_result_test, threshold_test, test_CI_L, test_CI_H  = auc_report(test_label, y_proba_test, out_path_test_clf)
                print("{} test set: auc = {}".format(moda_name_i,  auc_result_test))
                result_logger(out_path_test_clf, "test set: auc = {}\n".format(auc_result_test))
                test_acc, test_sen, test_spe, test_precision, test_f1 = analysis_report(y_test, y_pred_threshold,
                                                                                     out_path_test_clf, label_group_i,moda_name_i+"_"+model_name_i)
                # print(test_acc, test_sen, test_spe, test_CI_L, test_CI_H)


                data_result = [ran_statue, label_name, moda_name_i, model_name_i, len(X_train_sel.columns),
                               auc_result_train, tra_CI_L, tra_CI_H, tra_acc, tra_sen, tra_spe, tra_precision, tra_f1,
                               auc_result_test, test_CI_L, test_CI_H, test_acc, test_sen, test_spe, test_precision, test_f1]

                result.append(data_result)

    result = pd.DataFrame(result, columns=['ran_statue', 'exp_label_name', 'mod_name_i', 'model_name_i', 'num_feature',
                                           'roc_auc_macro_train', 'tra_CI_L', 'tra_CI_H',
                                           'tra_acc', 'tra_sen', 'tra_spe', 'tra_precision', 'tra_f1',
                                           'roc_auc_macro_test', 'test_CI_L', 'test_CI_H',
                                           'test_acc', 'test_sen', 'test_spe','test_precision', 'test_f1'])
    result.to_csv(os.path.join(out_path_level1, 'result_{}.csv'.format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()))), encoding='936')


def main():
    analysis_data()


if __name__ == '__main__':
    main()