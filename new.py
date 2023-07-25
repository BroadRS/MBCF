import csv
import os

import GetKNearestNeighbor
from evaluations import RMSE, RMSE2
from readerdata import *
from test import *
from bls_addenhencenodes import *
from itertools import chain
from numpy import genfromtxt
from sklearn import metrics
import scipy.io as scio
import numpy as np
import numpy as np
import scipy.io as scio
from BroadLearningSystem import BLS, BLS_AddEnhanceNodes, BLS_AddFeatureEnhanceNodes


if __name__ == "__main__":
    N1 = 10  # # of nodes belong to each window
    N2 = 10  # # of windows -------Feature mapping layer
    N3 = 500  # # of enhancement nodes -----Enhance layer
    L = 5  # # of incremental steps
    M1 = 50  # # of adding enhance nodes
    s = 0.8  # shrink coefficient
    C = 2 ** -30  # Regularization coefficient
    # k_users = [3,5,7,9,11,13] # the number of users' neighbors
    # k_items = [3,5,7,9,11,13] # the number of items' neighbors
    ar = 1
    # dataset = 'matrix_common_team_count'
    # lst_v = []
    # with open("matrix_common_team_count.csv", "r") as f:
    #     reader = csv.reader(f)
    #     lsts = []
    #     for line in reader:
    #         lst = []
    #         for i in range(len(line)):
    #             if (line[i] != ' ' and line[i] != '/n'):
    #                 lst.append(int(line[i]))
    #
    #         lsts.append(lst)
    # lst_v.append(lsts)
    # f.close()
    dataFile = ['transratings_Patio_Lawn_and_Garden.csv'
                ]
    map_nums = [5,10, 15,20,25]
    enhance_nums = [5,10,15,20,25]
    batchsizes = [6]
    k_users = [3,5,7,9,11]
    k_items = [3,5,7,9,11]
    EPOCH = 2
    for dataFile in dataFile:
        numberOfUser, numberOfItem, mtx_np =readamazon2(dataFile)
        neighbor_user = GetKNearestNeighbor.k_neighbors(mtx_np, int((numberOfUser-1)*0.1), numberOfUser)
        neighbor_item = GetKNearestNeighbor.k_neighbors(mtx_np.T, int((numberOfItem-1)*0.05), numberOfItem)
        print("data reader complete")
        for k_user in [10]:
            for k_item in [10]:
                traindata, testdata, trainlabel, testlabel = predict_user_and_item(numberOfUser,
                                                                                   numberOfItem,
                                                                                   mtx_np,
                                                                                   neighbor_user, k_user,
                                                                                   neighbor_item, k_item)
                trainlabel = trainlabel.flatten()
                testlabel = testlabel.flatten()
                for map_num in [15,25,30,35,40]:#25
                    for enhance_num in [30]:#25
                        for map_batchsize in [20]:#15
                                for enh_batchsize in [20]:#10
                                    print(traindata.shape)
                                    print(testdata.shape)
                                    print(trainlabel.shape)
                                    print(testlabel.shape)
                                    bls = broadNet(map_num=map_num,  # 初始时多少组mapping nodes
                                                   enhance_num=enhance_num,  # 初始时多少enhancement nodes
                                                   EPOCH=1,  # 训练多少轮
                                                   map_function='relu',
                                                   enhance_function='relu',
                                                   map_batchsize=map_batchsize,  # 每一组的神经元个数
                                                   enh_batchsize=enh_batchsize,
                                                   DESIRED_ACC=0.95,  # 期望达到的准确率
                                                   STEP=int(1)  # 一次增加多少组enhancement nodes
                                                   )
                                    labelunique = {}

                                    num = 0
                                    for i in range(len(trainlabel)):
                                        if trainlabel[i] not in labelunique:
                                            labelunique.update({trainlabel[i]:num})
                                            num+=1
                                    labels = sorted(labelunique)
                                    # print('-------------------BLS_BASE---------------------------')
                                    # BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
                                    # print('-------------------BLS_ENHANCE------------------------')
                                    # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
                                    # print('-------------------BLS_FEATURE&ENHANCE----------------')
                                    # M2 = 50  # # of adding feature mapping nodes
                                    # M3 = 50  # # of adding enhance nodes
                                    # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

                                    starttime = datetime.datetime.now()
                                    bls.fit(traindata, trainlabel)
                                    endtime = datetime.datetime.now()
                                    runtime = str((endtime - starttime).total_seconds())
                                    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

                                    pre = bls.predict(testdata)
                                    teststarttime = datetime.datetime.now()
                                    predictlabel = bls.weightPredict(testdata)
                                    testendtime = datetime.datetime.now()
                                    testtime = str((testendtime - teststarttime).total_seconds())

                                    # predictlabel - ranges[0]
                                    # print(metrics.mean_absolute_error(testlabel, predictlabel))
                                    lista =[]
                                    for i in range(len(pre)):
                                        lista.append(labels[pre[i]])
                                    mae = str(metrics.mean_absolute_error(testlabel, lista))
                                    rmse = str(RMSE2(pre, testlabel, labels))
                                    print(metrics.mean_absolute_error(testlabel, lista))
                                    print(RMSE2(pre, testlabel, labels))
                                    txtname = dataFile+'test.txt'
                                    txtname = str(map_num) + str(enhance_num ) +str(map_batchsize )+'-'+str(enh_batchsize)+str(k_user)+"_"+str(k_item)+ txtname
                                    f = open(txtname, 'w')
                                    f.write(runtime)
                                    f.write(" ")
                                    f.write(testtime)
                                    f.write('\n')
                                    f.write(mae+'\n'+rmse)
                                    f.write("pre GT \n")
                                    for i in range(len(lista)):
                                        tmp  = str(lista[i]) + ' ' + str(testlabel[i])
                                        f.write(tmp)
                                        f.write('\n')
                                    f.close()
                                    # label = [1,2,3,4,5]

                for map_num in [30]:#25
                    for enhance_num in [15,25,30,35,40]:#25
                        for map_batchsize in [20]:#15
                                for enh_batchsize in [20]:#10
                                    print(traindata.shape)
                                    print(testdata.shape)
                                    print(trainlabel.shape)
                                    print(testlabel.shape)
                                    bls = broadNet(map_num=map_num,  # 初始时多少组mapping nodes
                                                   enhance_num=enhance_num,  # 初始时多少enhancement nodes
                                                   EPOCH=1,  # 训练多少轮
                                                   map_function='relu',
                                                   enhance_function='relu',
                                                   map_batchsize=map_batchsize,  # 每一组的神经元个数
                                                   enh_batchsize=enh_batchsize,
                                                   DESIRED_ACC=0.95,  # 期望达到的准确率
                                                   STEP=int(1)  # 一次增加多少组enhancement nodes
                                                   )
                                    labelunique = {}

                                    num = 0
                                    for i in range(len(trainlabel)):
                                        if trainlabel[i] not in labelunique:
                                            labelunique.update({trainlabel[i]:num})
                                            num+=1
                                    labels = sorted(labelunique)
                                    # print('-------------------BLS_BASE---------------------------')
                                    # BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
                                    # print('-------------------BLS_ENHANCE------------------------')
                                    # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
                                    # print('-------------------BLS_FEATURE&ENHANCE----------------')
                                    # M2 = 50  # # of adding feature mapping nodes
                                    # M3 = 50  # # of adding enhance nodes
                                    # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

                                    starttime = datetime.datetime.now()
                                    bls.fit(traindata, trainlabel)
                                    endtime = datetime.datetime.now()
                                    runtime = str((endtime - starttime).total_seconds())
                                    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

                                    pre = bls.predict(testdata)
                                    teststarttime = datetime.datetime.now()
                                    predictlabel = bls.weightPredict(testdata)
                                    testendtime = datetime.datetime.now()
                                    testtime = str((testendtime - teststarttime).total_seconds())

                                    # predictlabel - ranges[0]
                                    # print(metrics.mean_absolute_error(testlabel, predictlabel))
                                    lista =[]
                                    for i in range(len(pre)):
                                        lista.append(labels[pre[i]])
                                    mae = str(metrics.mean_absolute_error(testlabel, lista))
                                    rmse = str(RMSE2(pre, testlabel, labels))
                                    print(metrics.mean_absolute_error(testlabel, lista))
                                    print(RMSE2(pre, testlabel, labels))
                                    txtname = dataFile+'test.txt'
                                    txtname = str(map_num) + str(enhance_num ) +str(map_batchsize )+'-'+str(enh_batchsize)+str(k_user)+"_"+str(k_item)+ txtname
                                    f = open(txtname, 'w')
                                    f.write(runtime)
                                    f.write(" ")
                                    f.write(testtime)
                                    f.write('\n')
                                    f.write(mae+'\n'+rmse)
                                    f.write("pre GT \n")
                                    for i in range(len(lista)):
                                        tmp  = str(lista[i]) + ' ' + str(testlabel[i])
                                        f.write(tmp)
                                        f.write('\n')
                                    f.close()
                                    # label = [1,2,3,4,5]
                for map_num in [30]:#25
                    for enhance_num in [30]:#25
                        for map_batchsize in [10,15,20,30,35]:#15
                                for enh_batchsize in [20]:#10
                                    print(traindata.shape)
                                    print(testdata.shape)
                                    print(trainlabel.shape)
                                    print(testlabel.shape)
                                    bls = broadNet(map_num=map_num,  # 初始时多少组mapping nodes
                                                   enhance_num=enhance_num,  # 初始时多少enhancement nodes
                                                   EPOCH=1,  # 训练多少轮
                                                   map_function='relu',
                                                   enhance_function='relu',
                                                   map_batchsize=map_batchsize,  # 每一组的神经元个数
                                                   enh_batchsize=enh_batchsize,
                                                   DESIRED_ACC=0.95,  # 期望达到的准确率
                                                   STEP=int(1)  # 一次增加多少组enhancement nodes
                                                   )
                                    labelunique = {}

                                    num = 0
                                    for i in range(len(trainlabel)):
                                        if trainlabel[i] not in labelunique:
                                            labelunique.update({trainlabel[i]:num})
                                            num+=1
                                    labels = sorted(labelunique)
                                    # print('-------------------BLS_BASE---------------------------')
                                    # BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
                                    # print('-------------------BLS_ENHANCE------------------------')
                                    # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
                                    # print('-------------------BLS_FEATURE&ENHANCE----------------')
                                    # M2 = 50  # # of adding feature mapping nodes
                                    # M3 = 50  # # of adding enhance nodes
                                    # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

                                    starttime = datetime.datetime.now()
                                    bls.fit(traindata, trainlabel)
                                    endtime = datetime.datetime.now()
                                    runtime = str((endtime - starttime).total_seconds())
                                    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

                                    pre = bls.predict(testdata)
                                    teststarttime = datetime.datetime.now()
                                    predictlabel = bls.weightPredict(testdata)
                                    testendtime = datetime.datetime.now()
                                    testtime = str((testendtime - teststarttime).total_seconds())

                                    # predictlabel - ranges[0]
                                    # print(metrics.mean_absolute_error(testlabel, predictlabel))
                                    lista =[]
                                    for i in range(len(pre)):
                                        lista.append(labels[pre[i]])
                                    mae = str(metrics.mean_absolute_error(testlabel, lista))
                                    rmse = str(RMSE2(pre, testlabel, labels))
                                    print(metrics.mean_absolute_error(testlabel, lista))
                                    print(RMSE2(pre, testlabel, labels))
                                    txtname = dataFile+'test.txt'
                                    txtname = str(map_num) + str(enhance_num ) +str(map_batchsize )+'-'+str(enh_batchsize)+str(k_user)+"_"+str(k_item)+ txtname
                                    f = open(txtname, 'w')
                                    f.write(runtime)
                                    f.write(" ")
                                    f.write(testtime)
                                    f.write('\n')
                                    f.write(mae+'\n'+rmse)
                                    f.write("pre GT \n")
                                    for i in range(len(lista)):
                                        tmp  = str(lista[i]) + ' ' + str(testlabel[i])
                                        f.write(tmp)
                                        f.write('\n')
                                    f.close()
                                    # label = [1,2,3,4,5]
                for map_num in [30]:#25
                    for enhance_num in [30]:#25
                        for map_batchsize in [20]:#15
                                for enh_batchsize in [10,15,20,30,35]:#10
                                    print(traindata.shape)
                                    print(testdata.shape)
                                    print(trainlabel.shape)
                                    print(testlabel.shape)
                                    bls = broadNet(map_num=map_num,  # 初始时多少组mapping nodes
                                                   enhance_num=enhance_num,  # 初始时多少enhancement nodes
                                                   EPOCH=1,  # 训练多少轮
                                                   map_function='relu',
                                                   enhance_function='relu',
                                                   map_batchsize=map_batchsize,  # 每一组的神经元个数
                                                   enh_batchsize=enh_batchsize,
                                                   DESIRED_ACC=0.95,  # 期望达到的准确率
                                                   STEP=int(1)  # 一次增加多少组enhancement nodes
                                                   )
                                    labelunique = {}

                                    num = 0
                                    for i in range(len(trainlabel)):
                                        if trainlabel[i] not in labelunique:
                                            labelunique.update({trainlabel[i]:num})
                                            num+=1
                                    labels = sorted(labelunique)
                                    # print('-------------------BLS_BASE---------------------------')
                                    # BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
                                    # print('-------------------BLS_ENHANCE------------------------')
                                    # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
                                    # print('-------------------BLS_FEATURE&ENHANCE----------------')
                                    # M2 = 50  # # of adding feature mapping nodes
                                    # M3 = 50  # # of adding enhance nodes
                                    # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

                                    starttime = datetime.datetime.now()
                                    bls.fit(traindata, trainlabel)
                                    endtime = datetime.datetime.now()
                                    runtime = str((endtime - starttime).total_seconds())
                                    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

                                    pre = bls.predict(testdata)
                                    teststarttime = datetime.datetime.now()
                                    predictlabel = bls.weightPredict(testdata)
                                    testendtime = datetime.datetime.now()
                                    testtime = str((testendtime - teststarttime).total_seconds())

                                    # predictlabel - ranges[0]
                                    # print(metrics.mean_absolute_error(testlabel, predictlabel))
                                    lista =[]
                                    for i in range(len(pre)):
                                        lista.append(labels[pre[i]])
                                    mae = str(metrics.mean_absolute_error(testlabel, lista))
                                    rmse = str(RMSE2(pre, testlabel, labels))
                                    print(metrics.mean_absolute_error(testlabel, lista))
                                    print(RMSE2(pre, testlabel, labels))
                                    txtname = dataFile+'test.txt'
                                    txtname = str(map_num) + str(enhance_num ) +str(map_batchsize )+'-'+str(enh_batchsize)+str(k_user)+"_"+str(k_item)+ txtname
                                    f = open(txtname, 'w')
                                    f.write(runtime)
                                    f.write(" ")
                                    f.write(testtime)
                                    f.write('\n')
                                    f.write(mae+'\n'+rmse)
                                    f.write("pre GT \n")
                                    for i in range(len(lista)):
                                        tmp  = str(lista[i]) + ' ' + str(testlabel[i])
                                        f.write(tmp)
                                        f.write('\n')
                                    f.close()
                                    # label = [1,2,3,4,5]
                for map_num in [15,25,30,35,40,50]:#25
                    for enhance_num in [30]:#25
                        for map_batchsize in [20]:#15
                                for enh_batchsize in [20]:#10
                                    print(traindata.shape)
                                    print(testdata.shape)
                                    print(trainlabel.shape)
                                    print(testlabel.shape)
                                    bls = broadNet(map_num=map_num,  # 初始时多少组mapping nodes
                                                   enhance_num=enhance_num,  # 初始时多少enhancement nodes
                                                   EPOCH=1,  # 训练多少轮
                                                   map_function='relu',
                                                   enhance_function='relu',
                                                   map_batchsize=map_batchsize,  # 每一组的神经元个数
                                                   enh_batchsize=enh_batchsize,
                                                   DESIRED_ACC=0.95,  # 期望达到的准确率
                                                   STEP=int(1)  # 一次增加多少组enhancement nodes
                                                   )
                                    labelunique = {}

                                    num = 0
                                    for i in range(len(trainlabel)):
                                        if trainlabel[i] not in labelunique:
                                            labelunique.update({trainlabel[i]:num})
                                            num+=1
                                    labels = sorted(labelunique)
                                    # print('-------------------BLS_BASE---------------------------')
                                    # BLS(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3)
                                    # print('-------------------BLS_ENHANCE------------------------')
                                    # BLS_AddEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1)
                                    # print('-------------------BLS_FEATURE&ENHANCE----------------')
                                    # M2 = 50  # # of adding feature mapping nodes
                                    # M3 = 50  # # of adding enhance nodes
                                    # BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1, M2, M3)

                                    starttime = datetime.datetime.now()
                                    bls.fit(traindata, trainlabel)
                                    endtime = datetime.datetime.now()
                                    runtime = str((endtime - starttime).total_seconds())
                                    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

                                    pre = bls.predict(testdata)
                                    teststarttime = datetime.datetime.now()
                                    predictlabel = bls.weightPredict(testdata)
                                    testendtime = datetime.datetime.now()
                                    testtime = str((testendtime - teststarttime).total_seconds())

                                    # predictlabel - ranges[0]
                                    # print(metrics.mean_absolute_error(testlabel, predictlabel))
                                    lista =[]
                                    for i in range(len(pre)):
                                        lista.append(labels[pre[i]])
                                    mae = str(metrics.mean_absolute_error(testlabel, lista))
                                    rmse = str(RMSE2(pre, testlabel, labels))
                                    print(metrics.mean_absolute_error(testlabel, lista))
                                    print(RMSE2(pre, testlabel, labels))
                                    txtname = dataFile+'test.txt'
                                    txtname = str(map_num) + str(enhance_num ) +str(map_batchsize )+'-'+str(enh_batchsize)+str(k_user)+"_"+str(k_item)+ txtname
                                    f = open(txtname, 'w')
                                    f.write(runtime)
                                    f.write(" ")
                                    f.write(testtime)
                                    f.write('\n')
                                    f.write(mae+'\n'+rmse)
                                    f.write("pre GT \n")
                                    for i in range(len(lista)):
                                        tmp  = str(lista[i]) + ' ' + str(testlabel[i])
                                        f.write(tmp)
                                        f.write('\n')
                                    f.close()
                                    # label = [1,2,3,4,5]