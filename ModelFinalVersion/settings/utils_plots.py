import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from settings import running_settings

matplotlib.use('TkAgg')
plt.rcParams.update(
    {'font.size': running_settings.plot_settings['font_size']})


def plot_history(hist, model, fig):
    # plotting model history
    if not fig:
        fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)

    ax = fig.add_subplot(312)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='test')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)

    return fig


def plot_predictions_perrecording(y_p, y_all_tt, y_all_p, y_test_matrix, save, fold):
    fig = plt.figure(figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(y_all_tt, 'o--', label='True distance')
    ax.plot(y_all_p, 'o--', label='Predicted distance')
    ax.set_xlabel('Recordings')
    ax.set_ylabel('Distance [m]')
    ax.set_title('Whole distance prediction')
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(abs(np.array(y_all_p) - np.array(y_all_tt)), 'o--')
    ax.set_xlabel('Recordings')
    ax.set_ylabel('Absolute error [m]')
    ax.set_title('Whole distance prediction')
    plt.show(block=False)

    ax = fig.add_subplot(223)
    ax.plot(y_test_matrix, 'o--', label='True distance')
    ax.plot(y_p, 'o--', label='Predicted distance')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance [m]')
    ax.set_title('Sample distance prediction')
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(np.array(y_p) - np.array(y_test_matrix)), 'o--')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Absolute error [m]')
    ax.set_title('Sample distance prediction')
    plt.show(block=False)
    plt.tight_layout()
    if save:
        if running_settings.settings['fit_model']:
            fig_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '_predictions.eps'
        else:
            fig_name = running_settings.settings['model_name_load'] + '_predictions.eps'
        plt.savefig(running_settings.settings['model_images_path'] + os.sep + fig_name, format='eps', dpi=1000)

    return None



def plot_predictions_perrecording1(y_p, y_all_tt, y_all_p, y_test_matrix, save, fold):
    fig = plt.figure(figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    diffs = abs(np.array(y_all_p) - np.array(y_all_tt))

    ax = fig.add_subplot(211)
    ax.plot(abs(np.array(y_p) - np.array(y_test_matrix)), 'o')
    ax.set_xlabel('5-second chunks')
    ax.set_ylabel('Absolute error [m]')
    ax.set_title('5-second chunk distance estimation')
    plt.show(block=False)


    ax = fig.add_subplot(212)
    ax.bar(range(len(diffs)), diffs, tick_label=[str(i + 1) for i in range(len(diffs))])
    ax.set_xlabel('Recordings')
    ax.set_ylabel('Absolute error [m]')
    ax.set_title('6-Minute distance estimation')
    plt.show(block=False)
    plt.tight_layout()

    if save:
        if running_settings.settings['fit_model']:
            fig_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '_predictions.eps'
        else:
            fig_name = running_settings.settings['model_name_load'] + '_predictions.eps'
        plt.savefig(running_settings.settings['model_images_path'] + os.sep + fig_name, format='eps', dpi=1000)

    return None