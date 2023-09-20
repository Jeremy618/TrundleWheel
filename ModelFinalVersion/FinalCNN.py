import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from settings import running_settings, utils_functions, utils_plots, utils_cnn_architecture
from settings.utils_functions import get_stats, prepare_matrixes

if __name__ == '__main__':
    # Loading data and normalization (if specified)
    all_data, walk_start, test_start, recordings_length = utils_functions.load_data()
    all_data, distance, rec_length, timing = utils_functions.normalize(all_data)

    # Separation of cross validation and test data
    primi6min = running_settings.settings['primi6min']
    X_test, y_test, rec_length_test, X_cv, y_cv, rec_length_cv = utils_functions.separate_test(all_data, distance,
                                                                                               rec_length,
                                                                                               recordings_length,
                                                                                               primi6min=primi6min,
                                                                                               nrecordings=len(
                                                                                                   walk_start),
                                                                                               timing=timing)

    # Fitting or loading an existing model
    if running_settings.settings['fit_model']:
        input_shape = (300, 6)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        best_mdl, fold, best_stats = utils_functions.cross_validation(X_cv, y_cv, rec_length_cv, cv, input_shape)
    else:
        best_mdl, fold = utils_functions.load_existing_model(running_settings.settings['model_path'],
                                                             running_settings.settings['model_name_load'])
        image_path = running_settings.settings['model_images_path'] + os.sep + running_settings.settings[
            'model_name_load'] + '_visual.eps'

    # Preparing data for holdout test estimation
    X_test_matrix, y_test_matrix, rec_belonging_test = prepare_matrixes(X_test, y_test, rec_length_test)
    X_test_matrix = np.array(X_test_matrix)
    y_test_matrix = np.array(y_test_matrix)

    # Plot example of a single 5-second chunk
    Xsingle = X_test_matrix[0]
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title('5-second chunk accelerometer')
    ax[0].plot(Xsingle[:, 0], 'o--', label='acc_x', markersize=3.5)
    ax[0].plot(Xsingle[:, 1], 'o--', label='acc_y', markersize=3.5)
    ax[0].plot(Xsingle[:, 2], 'o--', label='acc_z', markersize=3.5)
    ax[0].set_xlabel('Samples')
    ax[0].set_ylabel('Acceleration [m/$s^{2}$]')
    ax[0].legend(loc='upper right')
    ax[1].set_title('5-second chunk rotation rate')
    ax[1].plot(Xsingle[:, 3], 'o--', label='alpha', markersize=3.5)
    ax[1].plot(Xsingle[:, 4], 'o--', label='beta', markersize=3.5)
    ax[1].plot(Xsingle[:, 5], 'o--', label='gamma', markersize=3.5)
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel('Samples')
    ax[1].set_ylabel('Rotation rate [rad/s]')
    plt.tight_layout()
    plt.show()

    # Estimating walked distance on holdout data
    y_p, y_all_tt, y_all_p = utils_functions.holdouttest_prediction(best_mdl, X_test_matrix, y_test_matrix,
                                                                    rec_belonging_test)

    # Plotting predictions (4 plots)
    utils_plots.plot_predictions_perrecording(y_p, y_all_tt, y_all_p, y_test_matrix,
                                               save=False, fold=fold)
    # Plotting predictions (2 plots)
    utils_plots.plot_predictions_perrecording1(y_p, y_all_tt, y_all_p, y_test_matrix,
                                               save=False, fold=fold)

    # Computing statistics of holdout test result
    stats_whole = get_stats(error=np.array(y_all_tt) - np.array(y_all_p), name='', df_stats=pd.DataFrame(), gt=y_all_tt,
                            est=y_all_p)
    stats_whole.rename(columns={0: "whole_dist"}, inplace=True)
    stats_samples = get_stats(error=y_test_matrix - y_p, name='', df_stats=pd.DataFrame(), gt=y_test_matrix, est=y_p)
    stats_samples.rename(columns={0: "samples_dist"}, inplace=True)
    all_stats = pd.concat([stats_whole, stats_samples], axis=1)
    if running_settings.settings['fit_model']:
        best_stats.rename(columns={0: 'samples_dist_fold'}, inplace=True)
        all_stats = pd.concat([all_stats, best_stats], axis=1)

    # Saving stats into csv
    if running_settings.settings['stats_save']:
        model_path = running_settings.settings['model_images_path'] + os.sep + "stats"
        if running_settings.settings['fit_model']:
            stats_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '_stats.csv'
        else:
            stats_name = running_settings.settings['model_name_load'] + '_stats.csv'
        all_stats.to_csv(model_path + os.sep + stats_name, index=True)

    print('document end')
