import math
import os
import pickle

import numpy as np
import pandas as pd

from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from settings import running_settings, utils_cnn_architecture


def load_data():
    file_path = running_settings.MOTHERPATH + os.sep + "all_datas_2"
    all_data = pd.read_csv(file_path + '.csv', sep=';')

    # Find indexes of valid, train and test data in the "all_datas_2" file

    walk_start = [0]  # indexes where each walk begin
    test_start = []  # indexes where each test data begin (4 last minutes of each walk (= 1 hour))
    chunk_test = running_settings.settings['chunk_test']  # = 60*60*4 = 4 minutes
    recordings_length = []
    r = 0
    for i in range(len(all_data) - 1):
        current_value = all_data.iloc[i, 0]
        next_value = all_data.iloc[i + 1, 0]
        all_data.at[i, 'recording'] = r
        if current_value > next_value:
            r += 1
            walk_start.append(i + 1)
            test_start.append(i - (chunk_test))
            recordings_length.append(walk_start[-1] - walk_start[-2])
    all_data.at[i + 1, 'recording'] = r  # otherwise the last value of recording in all data would be nan
    # add the last walk index
    # walk_start.append(len(all_data)-1) # the last element is 716147, all data shape is (716148, 12)
    test_start.append(len(all_data) - (chunk_test) - 1)  # the last element is 701747 = 716148 - chunk_test - 1
    recordings_length.append(len(all_data) - walk_start[-1])
    print("walk_start = ", walk_start)
    print("test_start = ", test_start)

    return all_data, walk_start, test_start, recordings_length


def split_data(all_data, walk_start, test_start, recordings_length, distance, rec_length, timing):
    # Split the data:
    #  - Test data = 1 hour (4 mins of each walk)
    #  - Train data = 70% of remaining data
    #  - Valid data = 30% of remaining data
    rec_length = np.array(rec_length)
    # Columns 0 to 11 : [time, m_acc_x, m_acc_y, m_acc_z, m_accG_x, m_accG_y, m_accG_z, m_rotTate_alpha, m_rotTate_beta, m_rotTate_gamma, steps, distance]

    col_s = running_settings.settings['col_s']  # first column
    col_f = running_settings.settings['col_f']  # last column

    X_train_list = []
    Y_train_list = []
    X_valid_list = []
    Y_valid_list = []
    X_test_list = []
    Y_test_list = []
    X_t = []
    y_t = []
    X_v = []
    X_tt = []
    y_v = []
    y_tt = []
    train_lengths = []
    valid_lengths = []
    test_lengths = []
    # number of rows of train and valid data given to the algorithm at each iteration in the loop
    chunk = running_settings.settings['chunk']  # 300 = 60hz*5 secs (60hz sampling frequency of accelerometer)

    for walk_idx in range(len(walk_start)):
        # for train and valid data
        # start_idx = walk_start[walk_idx]
        start_idx = np.where(rec_length == 0)[0][0]
        end_idx = test_start[walk_idx]
        split_point = start_idx + int((end_idx - start_idx) * 0.7)  # index where train data stop and valid data start
        # Adjust split_point and end_idx because valid and train data must be divisible by chunks
        split_point = split_point - (split_point - start_idx) % chunk
        end_idx = end_idx - (end_idx - split_point) % chunk

        X_train_list.append(all_data[start_idx:split_point])
        Y_train_list.append(distance[start_idx:split_point])
        X_valid_list.append(all_data[split_point:end_idx])
        Y_valid_list.append(distance[split_point:end_idx])

        # for test data
        start_idx = test_start[walk_idx]
        # end_idx = walk_start[walk_idx + 1] # end_idx = end_idx +
        end_idx = walk_start[walk_idx] + recordings_length[walk_idx]
        X_test_list.append(all_data[start_idx:end_idx])
        Y_test_list.append(distance[start_idx:end_idx])

        prevlength = len(X_t)
        for g in range(0, X_train_list[-1].shape[0] - chunk, chunk):
            print(g)
            X_t.append(X_train_list[-1][g:g + chunk, :])
            y_t.append(Y_train_list[-1].to_list()[g + chunk - 1] - Y_train_list[-1].to_list()[g])
        train_lengths.append(len(X_t) - prevlength)

        prevlength = len(X_v)
        for g in range(0, X_valid_list[-1].shape[0] - chunk, chunk):
            X_v.append(X_valid_list[-1][g:g + chunk, :])
            y_v.append(Y_valid_list[-1].to_list()[g + chunk - 1] - Y_valid_list[-1].to_list()[g])
        valid_lengths.append(len(X_v) - prevlength)

        prevlength = len(X_tt)
        for g in range(0, X_test_list[-1].shape[0] - chunk, chunk):
            X_tt.append(X_test_list[-1][g:g + chunk, :])
            y_tt.append(Y_test_list[-1].to_list()[g + chunk - 1] - Y_test_list[-1].to_list()[g])
        test_lengths.append(len(X_tt) - prevlength)

    X_t = np.array(X_t)
    X_v = np.array(X_v)
    X_tt = np.array(X_tt)
    y_t = np.array(y_t)
    y_tt = np.array(y_tt)
    y_v = np.array(y_v)
    # print("X_train shape:", X_train.shape)
    # print("y_train shape:", y_train.shape)
    # print("X_valid shape:", X_valid.shape)
    # print("y_valid shape:", y_valid.shape)
    # print("X_test shape:", X_test.shape)
    # print("y_test shape:", y_test.shape)

    # y_tt = np.array(y_tt)

    # X_t = np.expand_dims(X_t, axis=0)
    # X_v = np.expand_dims(X_v, axis=0)
    # X_tt = np.expand_dims(X_tt, axis=0)

    #  we need input_shape = (6, 300, 1)
    # X_t = X_t.transpose((1, 3, 2, 0))
    # X_v = X_v.transpose((1, 3, 2, 0))
    # X_tt = X_tt.transpose((1, 3, 2, 0))

    input_shape = (chunk, col_f - col_s, chunk)

    return X_t, X_v, X_tt, y_t, y_v, y_tt, input_shape, train_lengths, valid_lengths, test_lengths


def fitting(model, X_train_normalized, y_train, X_valid_normalized, y_valid):
    # fix random seed for reproductability
    seed = 7
    np.random.seed(seed)
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, verbose=1,
    #                                restore_best_weights=True)
    # best_mdl_path = 'bst_mdl.h5'
    # mdl_checkpoint = ModelCheckpoint(best_mdl_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Fit the model
    X_train_normalized = np.squeeze(X_train_normalized)
    X_valid_normalized = np.squeeze(X_valid_normalized)
    X_train_normalized = X_train_normalized.transpose((0, 2, 1))
    X_valid_normalized = X_valid_normalized.transpose((0, 2, 1))
    batch_size = 32
    hist = model.fit(X_train_normalized, y_train, epochs=150, validation_data=(X_valid_normalized, y_valid),
                     batch_size=batch_size,
                     shuffle=False)

    return hist, model


def get_stats(error, name, df_stats, gt, est):
    errorperc = []
    for g, ground in enumerate(gt):
        errorperc.append(((abs(est[g] - ground)) / ground) * 100)

    print('#####################################################')
    print('Statistic for ', name)
    print('mean: ', np.mean(error))
    print('median: ', np.median(error))
    mse = np.square(error).mean()
    print('RMSE: ', math.sqrt(mse))
    print('SD: ', np.std(error))
    print('ABS median: ', np.median(abs(error)))
    print('ABS mean: ', np.mean(abs(error)))
    print('ABS SD: ', np.std(abs(error)))
    print('ABS min: ', min(abs(error)))
    print('ABS max: ', max(abs(error)))
    print('ABS mean: ', np.mean(abs(error)))
    print('IQR range: ', stats.iqr(abs(error)))
    print('Q1(abs): ', np.percentile(abs(error), 25))  # il BELOWPERC delle risposte sta sotto quest valore
    print('Q3(abs): ', np.percentile(abs(error), 75))
    print('Q1: ', np.percentile((error), 25))  # il BE# LOWPERC delle risposte sta sotto quest valore
    print('Q3: ', np.percentile((error), 75))
    print('Correlation GT - EST: ', np.corrcoef(gt, est)[0][1])
    # print('Correlation: ', stats.pearsonr(np.array(gt_dist), np.array(est_dist)))
    df_stats.at[0, 'mean'] = round(np.mean(error), 2)
    df_stats.at[0, 'median'] = round(np.median(error), 2)
    df_stats.at[0, 'mse'] = round(mse, 2)
    df_stats.at[0, 'RMSE'] = round(math.sqrt(mse), 2)
    df_stats.at[0, 'SD'] = round(np.std(error), 2)
    df_stats.at[0, 'ABS median'] = round(np.median(abs(error)), 2)
    df_stats.at[0, 'ABS mean'] = round(np.mean(abs(error)), 2)
    df_stats.at[0, 'ABS SD'] = round(np.std(abs(error)), 2)
    df_stats.at[0, 'ABS min'] = round(min(abs(error)), 2)
    df_stats.at[0, 'ABS max'] = round(max(abs(error)), 2)
    df_stats.at[0, 'ABS mean'] = round(np.mean(abs(error)), 2)
    df_stats.at[0, 'min'] = round(min(error), 2)
    df_stats.at[0, 'max'] = round(max(error), 2)
    df_stats.at[0, 'IQR(abs)'] = round(stats.iqr(abs(error)), 2)
    df_stats.at[0, 'Q1(abs)'] = round(np.percentile(abs(error), 25), 2)
    df_stats.at[0, 'Q3(abs)'] = round(np.percentile(abs(error), 75), 2)
    df_stats.at[0, 'IQR'] = round(stats.iqr(error), 2)
    df_stats.at[0, 'Q1'] = round(np.percentile((error), 25), 2)
    df_stats.at[0, 'Q3'] = round(np.percentile((error), 75), 2)
    df_stats.at[0, 'corr'] = round(np.corrcoef(gt, est)[0][1], 2)
    df_stats.at[0, 'MAPE'] = round(np.average(errorperc), 2)

    df_stats.at[0, 'loa_low'] = np.mean(error) - 1.96 * np.std(error)
    df_stats.at[0, 'loa_high'] = np.mean(error) + 1.96 * np.std(error)
    df_stats.at[0, 'loa_low%'] = np.mean(errorperc) - 1.96 * np.std(errorperc)
    df_stats.at[0, 'loa_high%'] = np.mean(errorperc) + 1.96 * np.std(errorperc)

    df_stats = df_stats.transpose()
    return df_stats


def holdouttest_prediction(model, X_tt, y_tt, test_lengths):
    # Prediction:

    nrecordings = len(np.unique(test_lengths))

    # X_tt = X_tt.transpose((0, 2, 1))
    y_p = model.predict(X_tt, verbose=True)[:, 0]

    y_all_p = []
    y_all_tt = []
    counter = 0
    for i in range(nrecordings):
        y_true_sum = 0
        y_pred_sum = 0
        onerecording = True
        while onerecording and counter < len(y_tt):
            if test_lengths[counter] == i:
                y_true_sum += y_tt[counter]
                y_pred_sum += y_p[counter]
                counter += 1
            else:
                onerecording = False
                print('Changing recording, recording:' + str(i) + ' - counter: ' + str(counter))
                print('Total distance true: ' + str(y_true_sum))
                print('Total distance pred: ' + str(y_pred_sum))
        y_all_tt.append(y_true_sum)
        y_all_p.append(y_pred_sum)

    return y_p, y_all_tt, y_all_p


def normalize(df_res):
    col_s = running_settings.settings['col_s']  # first column
    col_f = running_settings.settings['col_f']  # last column

    if running_settings.settings['resampling_data']:
        resampling_fs = 59
        timedelta_array = pd.to_timedelta(np.concatenate(df_res[['time']].to_numpy()), unit='ms')
        # df_res = df_res.resample(str(resampling_fs)+"S").ffill(limit=1).interpolate(numeric_only=False)
    if running_settings.settings['normalize_data'] == 'minmax':
        scaler = MinMaxScaler()
        all_data = scaler.fit_transform(df_res.iloc[:, col_s:col_f])
    else:
        all_data = np.array(df_res.iloc[:, col_s:col_f])
    distance = df_res.iloc[:, -2]

    return all_data, distance, df_res['recording'].to_list(), df_res.iloc[:, 0].to_list()


def separate_test(all_data, distance, rec_length, recording_lengths, primi6min, nrecordings, timing):
    # Separate the test data from the train data
    # Test data: 6 minutes from each walk either at the beginning or at the end of the recording
    X_cv = []
    X_test = []
    y_cv = []
    y_test = []
    rec_length_test = []
    rec_length_cv = []

    counter = 0
    for n in range(nrecordings):
        print('Changing recording: ' + str(n))
        onerecording = True
        time_sum = []
        start_recording = counter
        end_recording = start_recording + recording_lengths[n]
        end_counter = 0
        while onerecording:
            if rec_length[counter] != n:
                onerecording = False
            elif primi6min:
                onerecording = True
                time_sum.append(timing[counter])
                counter += 1
                if len(time_sum) > 1:
                    # from the first try delta_time comes 360119.0 so maybe time_sum corresponds to seconds*10-3
                    delta_time = time_sum[-1] - time_sum[0]
                    # delta_time = datetime.fromtimestamp(delta_time)
                    if delta_time > running_settings.settings['chunk_test_time']:  # 360000 milliseconds = 6 minutes
                        # if len(time_sum) > running_settings.settings['chunk_test']: # 21600
                        X_test.append(all_data[start_recording:counter, :])
                        y_test.append(distance[start_recording:counter])
                        rec_length_test.append(np.ones(counter - start_recording) * n)
                        X_cv.append(all_data[counter:end_recording, :])
                        y_cv.append(distance[counter:end_recording])
                        rec_length_cv.append(np.ones(end_recording - counter) * n)
                        counter = end_recording
                        onerecording = False
            elif not primi6min:
                onerecording = True
                time_sum.append(timing[end_recording - end_counter - 1 ])
                end_counter += 1
                counter += 1
                if len(time_sum) > 1:
                    delta_time = time_sum[0] - time_sum[-1]
                    if delta_time > running_settings.settings['chunk_test_time']:
                        X_test.append(all_data[start_recording + end_counter:end_recording, :])
                        y_test.append(distance[start_recording + end_counter:end_recording])
                        rec_length_test.append(np.ones(end_recording - end_counter) * n)
                        X_cv.append(all_data[start_recording:start_recording + end_counter, :])
                        y_cv.append(distance[start_recording:start_recording + end_counter])
                        rec_length_cv.append(np.ones(end_counter) * n)
                        counter = end_recording
                        onerecording = False

    test_perc = []
    sum_test = 0
    sum_cv = 0
    for i in range(15):
        sum_test+=X_test[i].shape[0]
        sum_cv+=X_cv[i].shape[0]
        test_perc.append(np.round(len(X_test[i])/(len(X_test[i])+len(X_cv[i]))*100, 2))
    print("Test percentages across 15 recordings: \n")
    print(test_perc)

    return X_test, y_test, rec_length_test, X_cv, y_cv, rec_length_cv


def prepare_matrixes(X, y, rec_length):
    print(1)
    together_x = np.concatenate(X)
    together_y = np.concatenate(y)
    together_lengths = np.concatenate(rec_length)
    chunk = running_settings.settings['chunk']
    X_matrix = []
    y_matrix = []
    rec_belonging = []
    for n in range(len(X)):
        consider_y = np.array(y[n])
        # prevlength = len(X_t)
        # for g in range(0, X_train_list[-1].shape[0] - chunk, chunk):
        #     print(g)
        #     X_t.append(X_train_list[-1][g:g + chunk, :])
        #     y_t.append(Y_train_list[-1].to_list()[g + chunk - 1] - Y_train_list[-1].to_list()[g])
        # train_lengths.append(len(X_t) - prevlength)

        for g in range(0, X[n].shape[0] - chunk, chunk):
            # print(g)
            X_matrix.append(X[n][g:g + chunk, :])
            y_matrix.append(consider_y[g + chunk - 1] - consider_y[g])
            rec_belonging.append(rec_length[n][g + chunk - 1])
    return X_matrix, y_matrix, rec_belonging


def choose_best_model(models, statistiche):
    mses = []
    for i in range(len(models)):
        mses.append(statistiche[i].loc['mse'].to_list()[0])
    print("Best fold: " + str(np.argmin(mses)) + " with mse: " + str(min(mses)) + " and mae: " + str(statistiche[np.argmin(mses)].loc['ABS mean'].to_list()[0]))
    best_model = models[np.argmin(mses)]
    fold = np.argmin(mses)
    if running_settings.settings['bst_model_save']:
        model_path = running_settings.settings['model_path']
        model_name = running_settings.settings['model_name_fit'] + '_'+str(np.argmin(mses))
        best_model.save(model_path + os.sep + model_name + '.h5')
    return best_model, fold, statistiche[np.argmin(mses)]


def cross_validation(X_cv, y_cv, rec_length_cv, cv, input_shape):

    X_matrixes, y_matrixes, rec_belonging = prepare_matrixes(X_cv, y_cv, rec_length_cv)
    X_matrixes = np.array(X_matrixes)
    y_matrixes = np.array(y_matrixes)

    models = []
    y_trues = []
    y_preds = []
    statistiche = []
    histories = []
    fold = 0

    # preparing histories plot
    fig_hist = plt.figure(figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax_mse_train = fig_hist.add_subplot(221)
    ax_mae_train = fig_hist.add_subplot(222)
    ax_mse_intest = fig_hist.add_subplot(223)
    ax_mae_intest = fig_hist.add_subplot(224)

    ax_mse_train.set_xlabel('Epoch')
    ax_mse_train.set_ylabel('MSE')
    ax_mae_train.set_xlabel('Epoch')
    ax_mae_train.set_ylabel('MAE')
    ax_mse_train.set_title('MSE - Train')
    ax_mae_train.set_title('MAE - Train')

    ax_mse_intest.set_xlabel('Epoch')
    ax_mse_intest.set_ylabel('MSE')
    ax_mae_intest.set_xlabel('Epoch')
    ax_mae_intest.set_ylabel('MAE')
    ax_mse_intest.set_title('MSE - Test')
    ax_mae_intest.set_title('MAE - Test')

    for train_index, test_index in cv.split(X_matrixes, y_matrixes):
        print('Running cross validation at fold: '+str(fold))
        # 66.66%, 33.33%
        x_train_fold = X_matrixes[train_index]
        x_test_fold = X_matrixes[test_index]
        y_train_fold = y_matrixes[train_index]
        y_test_fold = y_matrixes[test_index]

        # input_shape = (300, 6)
        # model, stats, y_p, y_tt, hist = utils_cnn_architecture.arc1D(input_shape, x_train_fold, y_train_fold,
        #                                                                       x_test_fold, y_test_fold, fold, ax_mse, ax_mae)


        model, stats, y_p, y_tt, hist = utils_cnn_architecture.arc1Dsim(input_shape, x_train_fold, y_train_fold,
                                                                              x_test_fold, y_test_fold, fold, ax_mse_train, ax_mae_train,
                                                                              ax_mse_intest, ax_mae_intest)

        # input_shape = (300, 6, 1)
        # model, stats, y_p, y_tt, hist = utils_cnn_architecture.arc2Dsim(input_shape, x_train_fold, y_train_fold,
        #                                                                       x_test_fold, y_test_fold, fold, ax_mse, ax_mae)

        # input_shape = (6, 300)
        # model, stats, y_p, y_tt, hist = utils_cnn_architecture.arc1DsimINV(input_shape, x_train_fold, y_train_fold,
        #                                                                       x_test_fold, y_test_fold, fold, ax_mse, ax_mae)

        # input_shape = (300, 6)
        # model, stats, y_p, y_tt, hist = utils_cnn_architecture.arc1Dlinear(input_shape, x_train_fold, y_train_fold,
        #                                                                       x_test_fold, y_test_fold, fold, ax_mse, ax_mae)

        fold+=1
        models.append(model)
        y_trues.append(y_tt)
        y_preds.append(y_p)
        statistiche.append(stats)
        histories.append(hist)

    ax_mse_train.legend()
    ax_mae_train.legend()
    ax_mse_intest.legend()
    ax_mae_intest.legend()
    plt.show(block=False)

    best_model, fold, best_stats = choose_best_model(models, statistiche)

    if running_settings.settings['plot_save']:
        history_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '_hist.eps'
        model_path = running_settings.settings['model_images_path']
        fig_hist.tight_layout()
        fig_hist.savefig(model_path + os.sep + history_name, format='eps', dpi=1000)

    return best_model, fold, best_stats


def load_existing_model(model_path, model_name):
    # loading model
    fold = model_name.split('_')[-1]
    return load_model(model_path + os.sep + model_name + '.h5'), fold


def optimize_epoch_batch(X_cv, y_cv, model):
    # define the grid search parameters
    batch_size = [16, 32, 64, 128, 256]
    epochs = [50, 100, 150]
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    scoring = 'neg_mean_squared_error'
    sklearnmodel = KerasRegressor(model=model, verbose=0)
    grid = GridSearchCV(estimator=sklearnmodel, param_grid=param_grid, n_jobs=-1, cv=3, scoring=scoring, verbose=1)
    grid_result = grid.fit(X_cv, y_cv)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    if running_settings.settings['save_bestparam']:
        # save parameters using pickle
        bestparam_path = running_settings.settings['bestparam_path']
        bestparam_name = running_settings.settings['bestparam_name']
        with open(bestparam_path + os.sep + bestparam_name + '.pkl', 'wb') as f:
            pickle.dump(grid_result.best_params_, f)

    return None


def load_bestparam():
    bestparam_path = running_settings.settings['bestparam_path']
    bestparam_name = running_settings.settings['bestparam_name']
    # loading pickeled file
    with open(bestparam_path + os.sep + bestparam_name + '.pkl', 'rb') as f:
        bestparam = pickle.load(f)
    return bestparam


def funsavefig(name):
    plt.savefig(running_settings.settings['model_images_path'] + os.sep + name, format='eps', dpi=1000)
    return None

