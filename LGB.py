import pandas as pd
import numpy as np
import time
import os
import lightgbm as lgb

ISR_min_th = 1e-3
loopTimes = 1
noise_power_real = pow(10, -127.33 / 10)  # Noise power in linear scale

# rough_lines = 8 * 8000
train_line_start = 5000
test_line_start = 250000

def fun(x, ys, yi, gamma, signal_length):
    return 10 * np.log10(np.dot(ys, np.power(10, x[:signal_length]))) \
        - 10 * np.log10(np.dot(yi, np.power(10, x[signal_length:])) + 1) - gamma

def processDataForLightGBM(rawData, inter_set, signal_set):
    omega_i = rawData.loc[:, inter_set]
    omega_s = rawData.loc[:, signal_set]
    gamma = rawData.iloc[:, -1]
    
    X = np.concatenate((omega_s, omega_i), axis=1)
    y = gamma
    return X, y

def custom_rmse(omega_signal, omega_inter, gamma, signal_length):
    def _custom_rmse(y_pred, train_data):
        residual = fun(y_pred[:PreNm], omega_signal, omega_inter, gamma, signal_length)
        grad = -2 * residual
        hess = np.ones_like(residual)
        return grad, hess
    return _custom_rmse

if __name__ == "__main__":
    # line = [8000, 20000, 40000, 80000, 160000, 320000, 640000]
    line = [10000, 40000, 160000, 640000]
    path_log = "./"
    timestamp = time.strftime("%y_%m_%d_%H_%M", time.localtime())
    path_time_no_recur = path_log + "Res/"+timestamp+"_LGBc/"
    if not os.path.exists(path_time_no_recur):
        os.makedirs(path_time_no_recur)

    f = open(path_time_no_recur + "parameters.txt", "w")
    x_scale = 1
    f.write("x_scale：" + str(x_scale) + "\n")
    f_scale = 1/2
    f.write("f_scale：" + str(f_scale) + "\n")
    ftol_rough = 1e-4
    f.write("rough setting:ftol  " + str(ftol_rough) +"\n")
    test_line = 20000
    f.write("Total test line :"+ str(test_line) + "start from" + str(test_line_start)+"\n")
    f.write("train line start"+ str(train_line_start)+"\n")
    ftol_fine = 1e-8
    f.write("ftol_fine"+str(ftol_fine)+"\n")
    f.write("max nfev inf!!!!!!!!!!")
    f.close()

    times_count = 0

    while times_count < loopTimes:
        times_count += 1
        path_time = path_time_no_recur + str(times_count) + "/"
        if not os.path.exists(path_time):
            os.makedirs(path_time)
        list_bm_rmse_lr = []
        list_t_lr = []
        n_count = []
        for multi_line_use in [10000]:  # [10000, 40000, 160000, 640000]
            list_bm_rmse_lr_t = []
            learning_time = 0.
            iterative_times = 0
            Bm = [15552]  # [15552, 15572, 18533, 21524, 15567]

            for BmI in Bm:
                BmNm = len(Bm)
                f = open(path_log + "Beam" + str(BmI) + "Pre.txt")
                lines = f.readlines()
                PreLin = np.array([eval(ele.strip("\n")) for ele in lines])
                SgnNm = sum(PreLin[:, 1] == BmI)
                Pre = PreLin[:, 2]
                PreNm = Pre.size
                Psi = Pre[0:SgnNm]
                Psi = np.power(10, Psi / 10)
                Pin = Pre[SgnNm:]
                Pin = np.power(10, Pin / 10)
                Pre = np.power(10, Pre / 10)
                f.close()

                signal_set = list(np.arange(SgnNm))
                inter_set = list(set(np.arange(PreNm)).difference(set(signal_set)))
                train_unorganized = []
                filename = path_log + "bm_log_" + str(BmI) + ".txt"
                train_line = pd.read_csv(filename, header=None, encoding='latin-1', engine='python')
                train_unorganized.append(train_line.loc[train_line_start+1:train_line_start+multi_line_use])
                train = pd.concat(train_unorganized)
                train.index = range(len(train))

                X, y = processDataForLightGBM(train, inter_set, signal_set)

                custom_params = {
                    'omega_signal': X[:, :SgnNm],
                    'omega_inter': X[:, SgnNm:],
                    'gamma': y,
                    'signal_length': SgnNm
                }

                print("Line",multi_line_use,"Beam",BmI, "  Training begin")
                start_train = time.perf_counter()
                lgb_train = lgb.Dataset(X, y)
                params = {
                    'objective': custom_rmse(custom_params['omega_signal'], custom_params['omega_inter'],
                                              custom_params['gamma'], custom_params['signal_length']),
                    'metric': 'None',  # 'rmse'
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'learning_rate': 0.1,
                    'min_data_in_leaf': 1,
                    'max_depth': PreNm - SgnNm,
                    'lambda_l1': 20,
                    'lambda_l2': 35
                }
                gbm = lgb.train(params, lgb_train, num_boost_round=100)
                train_time = time.perf_counter() - start_train
                print("Training end, time is :", train_time)
                learning_time += train_time

                test_start = time.perf_counter()
                print("testing Beam ", str(BmI), " **************")
                test_data = pd.read_csv(filename, header=None)
                test = test_data.loc[test_line_start+1:test_line_start + test_line]
                X_test, y_test = processDataForLightGBM(test, inter_set, signal_set)
                gamma_real = (np.dot(X_test[:, :SgnNm], Psi)) / \
                             (np.dot(X_test[:, SgnNm:], Pin) + noise_power_real)
                gamma_real = 10 * np.log10(gamma_real)

                PNR_pred = gbm.predict(X_test)
                PNR_pred = PNR_pred[:PreNm]
                SNR_pred = np.power(10, PNR_pred[:SgnNm])
                INR_pred = np.power(10, PNR_pred[SgnNm:])
                INR_pred_min = min(SNR_pred) * ISR_min_th
                INR_pred = np.clip(INR_pred, INR_pred_min, np.inf)

                gamma_pred = (np.dot(X_test[:, :SgnNm], SNR_pred)) / \
                             (np.dot(X_test[:, SgnNm:], INR_pred) + 1)
                gamma_pred = 10 * np.log10(gamma_pred)
                
                rmse_lr = np.sqrt(np.mean((gamma_real - gamma_pred) ** 2))
                if np.isnan(rmse_lr):
                    print("Error occurred! Check.")
                list_bm_rmse_lr_t.append(rmse_lr)
                test_time = time.perf_counter() - test_start
                print("********Total Test Time:", test_time)

                SNR_real = Psi/noise_power_real
                INR_real = Pin/noise_power_real
                PNR_real = np.concatenate((SNR_real, INR_real))
                PNR_real = np.log10(PNR_real)
                if BmI == 0:
                    f = open(path_time+"pre_res"+str(multi_line_use)+".txt", "w")
                else:
                    f = open(path_time+"pre_res"+str(multi_line_use)+".txt", "a")
                f.write("Beam"+str(BmI)+"pred:"+str(list(PNR_pred))+"\n")
                f.write("Beam"+str(BmI)+"real:"+str(list(PNR_real))+"\n")
                f.close()
        
            list_bm_rmse_lr.append(np.average(list_bm_rmse_lr_t))
            list_t_lr.append(learning_time)

            f = open(path_time + "learning_log.txt", "w")
            f.write("train lines ："+str(line)+"\n")
            f.write("SINR(DB) rmse error：" + str(list_bm_rmse_lr) + "\n")
            f.write("learning time：" + str(list_t_lr) + "\n")
            f.close()

            print("SINR(DB) rmse error Loop Record:"+str(list_bm_rmse_lr))
            print("learning time:" + str(list_t_lr))