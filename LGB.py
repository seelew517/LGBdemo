#coding:utf-8
from numpy.core.defchararray import array
import pandas as pd
import lightgbm as lgb
import math
import sys
import numpy as np
import time
import os

# clip_min_th = 10**3.3
ISR_min_th = 1e-3
loopTimes = 1
mimo_recv = 2
mimo_send = 2
noise_power_real = pow(10, -148.95 / 10)

# rough_lines = ue_number_each_enb * 2000
rough_lines = 8 * 8000
train_line_start = 5000
test_line_start = 250000

def fun(x, ys, yi, gamma, signal_length):
    # gamma为 10*lg 的db形式
    return 10 * np.log10(np.dot(ys, np.power(10, x[:signal_length]))) - 10 * np.log10(np.dot(yi, np.power(10, x[signal_length:])) + 1) - gamma
    # return 10 * np.log10(np.dot(ys, x[0:2])) - 10 * np.log10(np.dot(yi, x[2:])) - gamma

def processDataForOptimizer(rawData, inter_set, signal_set):
    omega_i = rawData.loc[:, inter_set]
    omega_s = rawData.loc[:, signal_set]
    gamma = rawData.loc[:, ue_number]
    return omega_s, omega_i, gamma

def custom_eval(preds, train_data):
    data = train_data.get_data()
    ys = data[:, :len(preds)]
    yi = data[:, len(preds):2*len(preds)]
    gamma = data[:, 2*len(preds):]
    signal_length = ys.shape[1]

    errors = []

    for i in range(len(preds)):
        pred = preds[i]
        gamma_i = gamma[i]
        ys_i = ys[i]
        yi_i = yi[i]
        error = fun(pred, ys_i, yi_i, gamma_i, signal_length)
        errors.append(error)

    rmse = np.sqrt(np.mean(np.square(errors)))

    return 'custom_rmse', rmse, False

# 2用户
if __name__ == "__main__":
    # 从interference.txt读取接收功率的理想值
    path_log = "../数据/Log-16-2/"
    f = open(path_log + "interference.txt")
    lines = f.readlines() #  lines以str的List,其元素为\n为标识结尾的str，即f的每一行
    inter_lines = lines[2::3]
    inter_lines = [eval(ele.strip("\n")) for ele in inter_lines] # 获取每一个用户的实际干扰值，并去掉换行符。eval函数将字符串str当成有效的表达式来求值并返回计算结果，
    power_lines = lines[1::3]
    power_lines = [float(ele.strip("\n")) for ele in power_lines]  # eNB RxPower in dB
    power_lines = [pow(10, ele / 10) for ele in power_lines]  # converted to real value
    f.close()

    # 获取仿真场景的参数设置
    ue_number = len(power_lines)
    ue_number_each_enb = ue_number - len(inter_lines[0].keys())
    enb_number = int(ue_number / ue_number_each_enb)
    ue_set = range(ue_number)
    enb_set = range(enb_number)
    inter_lines_enb = inter_lines[::ue_number_each_enb]

    # line = np.array([1000, 2000, 4000, 8000, 20000, 40000, 80000]) * ue_number_each_enb
    line = [8000, 20000, 40000, 80000, 160000, 320000, 640000]

    # 生成时间戳
    timestamp = time.strftime("%y_%m_%d_%H_%M", time.localtime())
    path_time_no_recur = path_log + "Res/" + timestamp + "_LGB/"
    if not os.path.exists(path_time_no_recur):
        os.makedirs(path_time_no_recur)

    # 记录solver参数
    f = open(path_time_no_recur + "parameters.txt", "w")
    # x_scale = 1
    # f.write("x_scale：" + str(x_scale) + "\n")
    # f_scale = 1 / 2
    # f.write("f_scale：" + str(f_scale) + "\n")
    # ftol_rough = 1e-4
    # f.write("rough setting:ftol  " + str(ftol_rough) + ",max lines  " + str(rough_lines) + "\n")
    # f.write("rough from huber to linear" + "\n")
    test_line = 20000
    f.write("Total test line :" + str(test_line) + "start from" + str(test_line_start) + "\n")
    f.write("train line start" + str(train_line_start) + "\n")
    # ftol_fine = 1e-8
    # f.write("ftol_fine" + str(ftol_fine) + "\n")
    # f.write("max nfev inf!!!!!!!!!!")
    f.close()

    error_recur = []
    learn_time_recur = []
    iter_time_recur = []
    times_count = 0
    # 循环loopTimes次，得到平均训练时间
    while times_count < loopTimes:
        times_count += 1
        path_time = path_time_no_recur + str(times_count) + "/"
        if not os.path.exists(path_time):
            os.makedirs(path_time)
        list_ue_rmse_lr = []
        list_t_lr = []
        n_count = []
        for multi_line_use in line:
            list_ue_rmse_lr_t = []
            learning_time = 0.
            iterative_times = 0
            # 训练过程
            for enb in range(enb_number):
                inter_set = list(inter_lines_enb[enb].keys())
                signal_set = list(set(ue_set).difference(set(inter_set)))
                train_unorganized = []
                train_rough_unorganized = []
                # train_line = int(multi_line_use)
                train_line_user = int(multi_line_use / ue_number_each_enb)
                rough_lines_user = int(rough_lines / ue_number_each_enb)
                # 将数据整理成以为基站为单位  
                for user in signal_set:
                    filename = path_log + "ul_log_" + str(user + int(ue_number / ue_number_each_enb) + 1) + ".txt"
                    train_single_user = pd.read_csv(filename, header=None)
                    train_unorganized.append(train_single_user.loc[train_line_start + 1:train_line_start + train_line_user])
                    train_line_rough = min(train_line_user, rough_lines_user)
                    train_rough_unorganized.append(train_single_user.loc[train_line_start + 1:train_line_start + train_line_rough])
                train = pd.concat(train_unorganized)
                train.index = range(len(train))
                train_rough = pd.concat(train_rough_unorganized)
                train_rough.index = range(len(train_rough))

                # 将原始数据转化为优化器需要的输入向量，分为训练集和测试集
                omega_signal_rough, omega_inter_rough, gamma_rough = processDataForOptimizer(train_rough, inter_set, signal_set)
                omega_signal, omega_inter, gamma = processDataForOptimizer(train, inter_set, signal_set)
                 # omega_signal_test, omega_inter_test, gamma_test = processDataForOptimizer(test, inter_set, signal_set)

                print(enb, "  Rough train begin")
                start_rough = time.perf_counter()

                train_data_rough = lgb.Dataset(np.hstack([omega_signal_rough, omega_inter_rough, gamma_rough.values.reshape(-1, 1)]), label=omega_signal_rough.values[:, 0])
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'verbosity': -1,
                    'num_leaves': 31,
                    'min_data_in_leaf': 1,
                    'max_depth': 30
                }
                model = lgb.train(params, train_data_rough, num_boost_round=100)
                rough_train_time_enb = time.perf_counter() - start_rough
                print("rough train end, time is:", rough_train_time_enb)
                learning_time += rough_train_time_enb

                print(enb, "  Fine tune begin")
                start = time.perf_counter()
                train_data = lgb.Dataset(np.hstack([omega_signal, omega_inter, gamma.values.reshape(-1, 1)]), label=omega_signal.values[:, 0])
                model = lgb.train(params, train_data, num_boost_round=100, init_model=model)
                fine_tune_time_enb = time.perf_counter() - start
                print("Fine tune end, time is:", fine_tune_time_enb)
                learning_time += fine_tune_time_enb
                iterative_times += 200

                signal_real = np.array([power_lines[e] for e in signal_set])
                inter_real = np.array(list(inter_lines_enb[enb].values()))
                
                # 测试过程（已知pred_res）
                test_start = time.perf_counter()
                for user in signal_set:
                    print("testing", user, "**************")
                    filename = path_log + "ul_log_" + str(user + int(ue_number / ue_number_each_enb) + 1) + ".txt"
                    data_single_user = pd.read_csv(filename, header=None)
                    test = data_single_user.loc[test_line_start + 1:test_line_start + test_line]
                    inter_set = list(inter_lines[user].keys())
                    signal_set = list(set(ue_set).difference(set(inter_set)))
                    omega_signal_test, omega_inter_test, gamma_test = processDataForOptimizer(test, inter_set, signal_set)
                    gamma_real = (np.dot(omega_signal_test, signal_real) * mimo_recv * mimo_send) / (np.dot(omega_inter_test, inter_real) * mimo_recv * mimo_send + noise_power_real)
                    gamma_real = 10 * np.log10(gamma_real)

                    gamma_pred = model.predict(np.hstack([omega_signal_test, omega_inter_test, gamma_test.values.reshape(-1, 1)]))
                    rmse_lr = np.sqrt(np.mean((gamma_real - gamma_pred) ** 2))
                    list_ue_rmse_lr_t.append(rmse_lr)
                    print("RMSE by LightGBM: ", rmse_lr)

                print(enb, "  test_end")
                test_end = time.perf_counter()
                print("test time is:", test_end - test_start)

                PNR_real = 10 * np.log10(np.concatenate((signal_real, inter_real)))

                if enb == 0:
                    f = open(path_time + "pre_res" + str(multi_line_use) + ".txt", "w")
                else:
                    f = open(path_time + "pre_res" + str(multi_line_use) + ".txt", "a")
                f.write("enb" + str(enb) + " pred:" + str(list(model.predict(np.hstack([omega_signal, omega_inter, gamma.values.reshape(-1, 1)])))) + "\n")
                f.write("enb" + str(enb) + " real:" + str(list(PNR_real)) + "\n")
                f.close()

            list_ue_rmse_lr.append(np.average(list_ue_rmse_lr_t))
            n_count.append(iterative_times / enb_number)
            list_t_lr.append(learning_time)

            f = open(path_time + "learning_log.txt", "w")
            f.write("n  error：" + str(list_ue_rmse_lr) + "\n")
            f.write("time：" + str(list_t_lr) + "\n")
            f.write("iterative times ：" + str(n_count) + "\n")
            f.write("train lines ：" + str(line) + "\n")
            f.close()
            print("LR:" + str(list_ue_rmse_lr))
            print("time:" + str(list_t_lr))

        error_recur.append(list_ue_rmse_lr)
        learn_time_recur.append(list_t_lr)
        iter_time_recur.append(n_count)

        list_error_recur = list(np.average(error_recur, axis=0))
        list_learn_time_recur = list(np.average(learn_time_recur, axis=0))
        list_iter_time_recur = list(np.average(iter_time_recur, axis=0))

        f = open(path_time_no_recur + "learning_log.txt", "w")
        f.write("n  error：" + str(list_error_recur) + "\n")
        f.write("time：" + str(list_learn_time_recur) + "\n")
        f.write("iterative times ：" + str(list_iter_time_recur) + "\n")
        f.write("train lines ：" + str(line) + "\n")
        f.close()

        f = open(path_time_no_recur + "learning_log_all.txt", "w")
        f.write("n  error：" + str(error_recur) + "\n")
        f.write("time：" + str(learn_time_recur) + "\n")
        f.write("iterative times ：" + str(iter_time_recur) + "\n")
        f.write("train lines ：" + str(line) + "\n")
        f.close()
