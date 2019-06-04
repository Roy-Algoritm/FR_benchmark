"""
Three metrics include AUC, ROC,
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import pylab as plt
import os
from  shutil import copy


def auc(df, need_plt=False):
    true_dict = []
    top1_success = []
    top1_failed = []
    false_dict = []
    err_dict = []
    success_search_time = 0
    error_search_time = 0

    top1_t = dict()
    for i in range(90,100,1):
        top1_t[i] = dict()
        top1_t[i][True] = list()
        top1_t[i][False] = list()

    total_score = dict()
    total_score[True] = list()
    total_score[False] = list()
    y_true = []
    y_score = []

    # get error pairs
    mismatch_dict = dict()
    top1_mismatch_count = 0
    top2plus_mismatch_count = 0
    base_face_path = "/Users/royalli/Dataset/LFW/lfw-org/LFW_benchmark_dataset/base_face"
    query_face_path = "/Users/royalli/Dataset/LFW/lfw-org/LFW_benchmark_dataset/query_face"
    mismatch_path = "/Users/royalli/Dataset/LFW/lfw-org/LFW_benchmark_dataset/mismatch"

    for row_index, row in df.iterrows():
        label = row[0]
        success_search_time += 1
        for col_index in range(1, 20, 2):
            _y = row[col_index]
            _prob = row[col_index + 1]

            if isinstance(_y, str) and isinstance(_prob, str):
                y = _y.replace('(', '')
                prob = float(_prob.replace(')', ''))
            else:
                success_search_time -= 1
                error_search_time += 1
                err_dict.append(label)
                print(label.split(' ')[0])
                break

            # label compare
            if y in label:
                # auc y_true and y_score for metrics.roc_curve()
                y_true.append(True)
                y_score.append(prob)
                total_score[True].append(prob)

                pair = [True, y, prob]
                true_dict.append(pair)
                if col_index == 1:
                    top1_success.append([label, y, prob])
                    for i in range(90,100,1):
                        if prob > i:
                            top1_t[i][True].append([label, y, prob])
            else:
                # auc y_true and y_score for metrics.roc_curve()
                y_true.append(False)
                y_score.append(prob)
                total_score[False].append(prob)

                if col_index == 1:
                    top1_failed.append([label, y, prob])
                    for i in range(90,100,1):
                        if prob > i:
                            top1_t[i][False].append([label, y, prob])

                if col_index == 1:
                    if prob > 90:
                        top1_mismatch_count += 1
                        sub_path = os.path.join(mismatch_path, 'top1', str(prob)+ str(top1_mismatch_count))
                        # print("mismatch label {} score {} -> {} , {}".format(top1_mismatch_count, prob, label, y))
                        base_face_abs_path = os.path.join(base_face_path, y)
                        pic_N = os.listdir(base_face_abs_path)[0]
                        base_face_abs_path = os.path.join(base_face_abs_path, pic_N)
                        # print(base_face_abs_path)
                        # assert os.path.exists(base_face_abs_path)
                        _y = label.split("_")
                        query_face = "_".join(_y[:-1])
                        query_face_abs_path = os.path.join(query_face_path, query_face, label)
                        # print(query_face_abs_path)
                        assert  os.path.exists(query_face_abs_path)
                        if not os.path.exists(sub_path):
                            os.mkdir(sub_path)
                        copy(base_face_abs_path, sub_path)
                        copy(query_face_abs_path, sub_path)

                else:
                    if prob > 90:
                        top2plus_mismatch_count +=1
                        sub_path = os.path.join(mismatch_path, 'top2plus', str(prob)+ str(top2plus_mismatch_count))
                        # print("mismatch label {} score {} -> {} , {}".format(top1_mismatch_count, prob, label, y))
                        base_face_abs_path = os.path.join(base_face_path, y)
                        pic_N = os.listdir(base_face_abs_path)[0]
                        base_face_abs_path = os.path.join(base_face_abs_path, pic_N)
                        # print(base_face_abs_path)
                        # assert os.path.exists(base_face_abs_path)
                        _y = label.split("_")
                        query_face = "_".join(_y[:-1])
                        query_face_abs_path = os.path.join(query_face_path, query_face, label)
                        print(query_face_abs_path)
                        assert  os.path.exists(query_face_abs_path)
                        if not os.path.exists(sub_path):
                            os.mkdir(sub_path)
                        copy(base_face_abs_path, sub_path)
                        copy(query_face_abs_path, sub_path)

                pair = [False, y, prob]
                false_dict.append(pair)

    print("Top1 Mismatch above 90 : {}".format(top1_mismatch_count))
    print("Total Mismatch above 90 : {}".format(top1_mismatch_count+ top2plus_mismatch_count))

    print("\n| Total search : {} | Search Time : {} | Error Time : {} |"
          .format(row_index + 1, success_search_time, error_search_time))
    print("".format())
    print("True Face {}".format(len(true_dict)))
    print("False Face {}".format(len(false_dict)))
    print("Top1 Success {}".format(len(top1_success)))
    print("Top1 Failed {}".format(len(top1_failed)))
    print("Top1 Rate {}".format(len(top1_success) / (len(top1_failed) + len(top1_success))))

    print("\n| Top1 Total ：{} | Top1 Success : {} | Top1 Failed : {} | Top Rate : {} | "
          .format(success_search_time, len(top1_success), len(top1_failed),
                  len(top1_success) / (len(top1_failed) + len(top1_success))))

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=True)
    roc_auc = metrics.auc(fpr, tpr)

    print("\nAUC Score is {}".format(roc_auc))

    if need_plt:
        plt.figure(figsize=(5, 5))
        plt.grid(linestyle='--', linewidth=1)
        plt.title(' HUAWEI ROC ON LFW ')
        plt.plot(fpr, tpr, 'b', label='Val AUC = %0.9f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    print("\nTop1 Failed dict {}".format(top1_failed))

    for i in range(90, 100, 1):
        print("Top1: threshold [{}] (True,False,Total):({},{},{})".format(i, len(top1_t[i][True]), len(top1_t[i][False]), len(top1_t[i][True])+len(top1_t[i][False])))
        # for item in top1_t[i][False]:
        #     print(item[0])

    print("阈值,识别成功,识别失败,总量,召回率,误识率,拒识率")
    for i in range(90, 100, 1):
        print("{},{},{},{},{},{},{}".format(
            i,
            len(top1_t[i][True]),
            len(top1_t[i][False]),
            (len(top1_t[i][True])+len(top1_t[i][False])),
            len(top1_t[i][True])/len(true_dict),
            len(top1_t[i][False])/(len(top1_t[i][True])*19),
            (len(true_dict) - len(top1_t[i][True]))/len(true_dict)
        )
        )


if __name__ == '__main__':
    df = pd.read_csv("./searchFace.log")
    auc(df)
