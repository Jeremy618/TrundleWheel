import numpy as np
import pandas as pd
from keras import Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Input, Conv2D, MaxPooling2D

from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt

from settings import utils_functions, running_settings

# Used architecture: arc1Dsim()

def arc1Dsim(input_shape, x_t, y_t, x_tt, y_tt, fold, ax_mse, ax_mae, ax_mse_test, ax_mae_test):
    # arc1DSIMPLE
    # Define CNN parameters:

    # fit model
    # model_path = running_settings.settings['model_path']
    # model_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '.h5'
    # mdl_checkpoint = ModelCheckpoint(model_path + os.sep + model_name, monitor='val_loss', mode='min', verbose=1,
    #                                  save_best_only=True)

    network_settings = running_settings.network_settings

    input_acc = Input(input_shape)                                          # (300,6)
    conv1 = Conv1D(64, 32, activation="relu", padding='same')(input_acc)    # shape: Nx300x64
    batch1 = BatchNormalization()(conv1)                                    # shape: Nx300x64
    max1 = MaxPooling1D(pool_size=3, padding='same')(batch1)                # shape: Nx100x64
    drop1 = Dropout(0.1)(max1)                                              # shape: Nx100x64

    conv2 = Conv1D(32, 16, activation="relu", padding='same')(drop1)        # shape: Nx100x32
    batch2 = BatchNormalization()(conv2)                                    # shape: Nx100x32
    max2 = MaxPooling1D(pool_size=3, padding='same')(batch2)                # shape: Nx34x32
    drop2 = Dropout(0.1)(max2)                                              # shape: Nx34x32

    conv3 = Conv1D(8, 3, activation="relu", padding='same')(drop2)          # shape Nx34x8
    batch3 = BatchNormalization()(conv3)                                    # shape Nx34x8
    max3 = MaxPooling1D(pool_size=3, padding='same')(batch3)                # shape Nx12x8

    flatten_final = Flatten()(max3)                                         # shape Nx96
    output = Dense(1, activation='relu')(flatten_final)                     # shape Nx1

    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # reproducibility
    seed = 7
    np.random.seed(seed)

    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=network_settings['patience'],
                                   verbose=1,
                                   restore_best_weights=True)

    hist = model.fit(x_t, y_t, epochs=network_settings['epochs'],
                     validation_split=network_settings['validation_split'],
                     batch_size=network_settings['batch_size'],
                     callbacks=[early_stopping], shuffle=False)

    ax_mse_test.plot(hist.history['val_loss'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mae_test.plot(hist.history['val_mae'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mse.plot(hist.history['loss'], 'o--', label='Train, fold: ' + str(fold))
    ax_mae.plot(hist.history['mae'], 'o--', label='Train, fold: ' + str(fold))

    # plot history
    fig = plt.figure(
        figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction, fold: ' + str(fold))
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # # Prediction
    print("Predicting on inner-test data: \n")
    y_p = model.predict(x_tt, verbose=True)[:, 0]
    stats = utils_functions.get_stats(error=y_tt - y_p, name='', df_stats=pd.DataFrame(), gt=y_tt, est=y_p)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p, 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction, fold: ' + str(fold))
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model, stats, y_p, y_tt, hist

"""
def channel1(dense):
    conv1 = Conv1D(64, 32, activation="relu", padding='same')(dense) # shape: Nx300x64
    batch1 = BatchNormalization()(conv1) # shape: Nx300x64
    max1 = MaxPooling1D(pool_size=3, padding='same')(batch1) # shape: Nx100x64
    drop1 = Dropout(0.1)(max1) # shape: Nx100x64

    conv2 = Conv1D(32, 16, activation="relu", padding='same', input_shape=(96, 3))(drop1) # shape: Nx100x32
    batch2 = BatchNormalization()(conv2) # shape: Nx100x32
    max2 = MaxPooling1D(pool_size=3, padding='same')(batch2) # shape: Nx34x32
    drop2 = Dropout(0.1)(max2) # shape: Nx34x32
    conv3 = Conv1D(16, 8, activation="relu", padding='same', input_shape=(96, 3))(drop2) # shape: Nx34x16
    res = Dropout(0.1)(conv3)
    return res


def model_architecture1D(input_shape, x_t, y_t, x_v, y_v, X_tt, y_tt):
    # Define CNN parameters:

    input_acc = Input((300, 6))
    dense = Dense(12, activation='relu')(input_acc)
    res_1 = channel1(dense)
    res_2 = channel1(dense)
    concat = Concatenate()([res_1, res_2])
    max_end = MaxPooling1D(pool_size=3)(concat)
    dense_final = Dense(12, activation='relu')(max_end)
    flatten_final = Flatten()(dense_final)
    output = Dense(1, activation='relu')(flatten_final)

    # output = Dense(1, activation='relu')(dense)
    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # fix random seed for reproductability
    seed = 7
    np.random.seed(seed)
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, verbose=1,
    #                                restore_best_weights=True)
    # best_mdl_path = 'bst_mdl.h5'
    # mdl_checkpoint = ModelCheckpoint(best_mdl_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # # Fit the model
    # X_train_normalized = np.squeeze(x_t)
    # X_valid_normalized = np.squeeze(x_v)
    # X_test_normalized = np.squeeze(X_tt)
    # X_train_normalized = X_train_normalized.transpose((0, 2, 1))
    # X_valid_normalized = X_valid_normalized.transpose((0, 2, 1))
    # X_test_normalized = X_test_normalized.transpose((0, 2, 1))

    # batch_size = 16
    # hist = model.fit(X_train_normalized, y_t, epochs=50, validation_data=(X_valid_normalized, y_v),
    #                  batch_size=batch_size,
    #                  shuffle=False)
    batch_size = 16
    hist = model.fit(x_t, y_t, epochs=50, validation_data=(x_v, y_v),
                     batch_size=batch_size,
                     shuffle=False)
    # Prediction
    y_p = model.predict(X_tt, verbose=True)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction')
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction')
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p[:, 0], 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction')
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p[:, 0] - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction')
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model


def model_architecture1D_simple(input_shape, x_t, y_t, x_v, y_v, X_tt, y_tt):
    # Define CNN parameters:

    input_acc = Input((300, 6))
    dense = Dense(12, activation='relu')(input_acc)
    max_end = MaxPooling1D(pool_size=3)(dense)
    flatten_final = Flatten()(max_end)
    output = Dense(1, activation='relu')(flatten_final)

    # output = Dense(1, activation='relu')(dense)
    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # fix random seed for reproductability
    seed = 7
    np.random.seed(seed)
    # early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=10, verbose=1,
    #                                restore_best_weights=True)
    # best_mdl_path = 'bst_mdl.h5'
    # mdl_checkpoint = ModelCheckpoint(best_mdl_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # # Fit the model
    # X_train_normalized = np.squeeze(x_t)
    # X_valid_normalized = np.squeeze(x_v)
    # X_test_normalized = np.squeeze(X_tt)
    # X_train_normalized = X_train_normalized.transpose((0, 2, 1))
    # X_valid_normalized = X_valid_normalized.transpose((0, 2, 1))
    # X_test_normalized = X_test_normalized.transpose((0, 2, 1))

    # batch_size = 16
    # hist = model.fit(X_train_normalized, y_t, epochs=50, validation_data=(X_valid_normalized, y_v),
    #                  batch_size=batch_size,
    #                  shuffle=False)
    batch_size = 16
    hist = model.fit(x_t, y_t, epochs=50, validation_data=(x_v, y_v),
                     batch_size=batch_size,
                     shuffle=False)
    # Prediction
    y_p = model.predict(X_tt, verbose=True)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction')
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction')
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p[:, 0], 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction')
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p[:, 0] - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction')
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model


def arc1D(input_shape, x_t, y_t, x_tt, y_tt, fold, ax_mse, ax_mae):
    # Define CNN parameters:

    network_settings = running_settings.network_settings

    input_acc = Input(input_shape)  # (300,6)
    dense = Dense(12, activation='relu')(input_acc)  # shape Nx300x12
    res_1 = channel1(dense)  # shape Nx34x16
    res_2 = channel1(dense)  # shape Nx34x16
    concat = Concatenate()([res_1, res_2])  # shape Nx34x32
    max_end = MaxPooling1D(pool_size=3)(concat)  # shape Nx11x32
    flatten_final = Flatten()(max_end)  # shape Nx352
    output = Dense(1, activation='relu')(flatten_final)  # shape Nx1

    # output = Dense(1, activation='relu')(dense)
    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # reproducibility
    seed = 7
    np.random.seed(seed)

    # fit model
    # model_path = running_settings.settings['model_path']
    # model_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '.h5'
    # mdl_checkpoint = ModelCheckpoint(model_path + os.sep + model_name, monitor='val_loss', mode='min', verbose=1,
    #                                  save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=network_settings['patience'],
                                   verbose=1,
                                   restore_best_weights=True)

    hist = model.fit(x_t, y_t, epochs=network_settings['epochs'], validation_split=network_settings['validation_split'],
                     batch_size=network_settings['batch_size'], callbacks=[early_stopping], shuffle=False)

    ax_mse.plot(hist.history['val_loss'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mae.plot(hist.history['val_mae'], 'o--', label='Validation, fold: ' + str(fold))

    # plot history
    fig = plt.figure(
        figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction, fold: ' + str(fold))
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # # Prediction
    print("Predicting on inner-test data: \n")
    y_p = model.predict(x_tt, verbose=True)[:, 0]
    stats = utils_functions.get_stats(error=y_tt - y_p, name='', df_stats=pd.DataFrame(), gt=y_tt, est=y_p)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p, 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction, fold: ' + str(fold))
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model, stats, y_p, y_tt, hist


def arc1Dlinear(input_shape, x_t, y_t, x_tt, y_tt, fold, ax_mse, ax_mae):
    # arc1DSIMPLE
    # Define CNN parameters:

    network_settings = running_settings.network_settings

    input_acc = Input(input_shape)                                          # (300,6)
    flatten_final = Flatten()(input_acc)                                         # shape Nx96
    conv1 = Conv1D(64, 32, activation="relu", padding='same')(flatten_final)    # shape: Nx300x64
    batch1 = BatchNormalization()(conv1)                                    # shape: Nx300x64
    max1 = MaxPooling1D(pool_size=3, padding='same')(batch1)                # shape: Nx100x64
    drop1 = Dropout(0.1)(max1)                                              # shape: Nx100x64
    conv2 = Conv1D(32, 16, activation="relu", padding='same')(drop1)        # shape: Nx100x32
    batch2 = BatchNormalization()(conv2)                                    # shape: Nx100x32
    max2 = MaxPooling1D(pool_size=3, padding='same')(batch2)                # shape: Nx34x32
    drop2 = Dropout(0.1)(max2)                                              # shape: Nx34x32
    conv3 = Conv1D(8, 3, activation="relu", padding='same')(drop2)          # shape Nx34x8
    batch3 = BatchNormalization()(conv3)                                    # shape Nx34x8
    max3 = MaxPooling1D(pool_size=3, padding='same')(batch3)                # shape Nx12x8
    output = Dense(1, activation='relu')(flatten_final)  # shape Nx1

    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # reproducibility
    seed = 7
    np.random.seed(seed)

    # fit model
    # model_path = running_settings.settings['model_path']
    # model_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '.h5'
    # mdl_checkpoint = ModelCheckpoint(model_path + os.sep + model_name, monitor='val_loss', mode='min', verbose=1,
    #                                  save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=network_settings['patience'],
                                   verbose=1,
                                   restore_best_weights=True)

    hist = model.fit(x_t, y_t, epochs=network_settings['epochs'], validation_split=network_settings['validation_split'],
                     batch_size=network_settings['batch_size'], callbacks=[early_stopping], shuffle=False)

    ax_mse.plot(hist.history['val_loss'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mae.plot(hist.history['val_mae'], 'o--', label='Validation, fold: ' + str(fold))

    # plot history
    fig = plt.figure(
        figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction, fold: ' + str(fold))
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # # Prediction
    print("Predicting on inner-test data: \n")
    y_p = model.predict(x_tt, verbose=True)[:, 0]
    stats = utils_functions.get_stats(error=y_tt - y_p, name='', df_stats=pd.DataFrame(), gt=y_tt, est=y_p)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p, 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction, fold: ' + str(fold))
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model, stats, y_p, y_tt, hist


def arc1DsimINV(input_shape, x_t, y_t, x_tt, y_tt, fold, ax_mse, ax_mae):
    # arc1DSIMPLE
    # Define CNN parameters:

    network_settings = running_settings.network_settings

    x_t = x_t.transpose((0, 2, 1))
    x_tt = x_tt.transpose((0, 2, 1))

    input_acc = Input(input_shape)  # (6, 300)
    conv1 = Conv1D(128, 32, activation="relu", padding='same')(input_acc)  # shape: Nx6x128
    batch1 = BatchNormalization()(conv1)  # shape: Nx6x128
    drop1 = Dropout(0.1)(batch1)  # shape:  Nx6x128
    conv2 = Conv1D(64, 16, activation="relu", padding='same')(drop1)  # shape: Nx6x64
    batch2 = BatchNormalization()(conv2)  # shape: Nx6x64
    drop2 = Dropout(0.1)(batch2)  # shape: Nx6x64
    conv3 = Conv1D(32, 8, activation="relu", padding='same')(drop2)     # shape: Nx6x32
    batch3 = BatchNormalization()(conv3) # shape Nx6x32
    max3 = MaxPooling1D(pool_size=3, padding='same')(batch3) # shape Nx2x32
    flatten_final = Flatten()(max3)  # shape Nx64
    output = Dense(1, activation='relu')(flatten_final)  # shape Nx1

    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # reproducibility
    seed = 7
    np.random.seed(seed)

    # fit model
    # model_path = running_settings.settings['model_path']
    # model_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '.h5'
    # mdl_checkpoint = ModelCheckpoint(model_path + os.sep + model_name, monitor='val_loss', mode='min', verbose=1,
    #                                  save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=network_settings['patience'],
                                   verbose=1,
                                   restore_best_weights=True)

    hist = model.fit(x_t, y_t, epochs=network_settings['epochs'], validation_split=network_settings['validation_split'],
                     batch_size=network_settings['batch_size'], callbacks=[early_stopping], shuffle=False)

    ax_mse.plot(hist.history['val_loss'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mae.plot(hist.history['val_mae'], 'o--', label='Validation, fold: ' + str(fold))

    # plot history
    fig = plt.figure(
        figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction, fold: ' + str(fold))
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # # Prediction
    print("Predicting on inner-test data: \n")
    y_p = model.predict(x_tt, verbose=True)[:, 0]
    stats = utils_functions.get_stats(error=y_tt - y_p, name='', df_stats=pd.DataFrame(), gt=y_tt, est=y_p)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p, 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction, fold: ' + str(fold))
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model, stats, y_p, y_tt, hist

def arc2Dsim(input_shape, x_t, y_t, x_tt, y_tt, fold, ax_mse, ax_mae):
    # arc1DSIMPLE
    # Define CNN parameters:

    network_settings = running_settings.network_settings

    input_acc = Input(input_shape)  # (300, 6, 1)
    conv1 = Conv2D(16, 8, activation="relu", padding='same')(input_acc)         # shape: Nx300x64
    batch1 = BatchNormalization()(conv1)                                        # shape: Nx300x64
    max1 = MaxPooling2D(pool_size=(3, 3), padding='same')(batch1)               # shape: Nx100x64
    drop1 = Dropout(0.1)(max1)                                                  # shape: Nx100x64
    conv2 = Conv1D(16, 8, activation="relu", padding='same')(drop1)             # shape: Nx100x32
    batch2 = BatchNormalization()(conv2)                                        # shape: Nx100x32
    max2 = MaxPooling2D(pool_size=(3, 3), padding='same')(batch2)               # shape: Nx34x32
    drop2 = Dropout(0.1)(max2)  # shape: Nx34x32
    conv3 = Conv2D(8, 3, activation="relu", padding='same')(drop2)      # shape Nx34x8
    batch3 = BatchNormalization()(conv3) # shape Nx34x8
    max3 = MaxPooling2D(pool_size=(3, 3), padding='same')(batch3) # shape Nx12x8
    flatten_final = Flatten()(max3)  # shape Nx96
    output = Dense(1, activation='relu')(flatten_final)  # shape Nx1

    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # reproducibility
    seed = 7
    np.random.seed(seed)

    # fit model
    # model_path = running_settings.settings['model_path']
    # model_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '.h5'
    # mdl_checkpoint = ModelCheckpoint(model_path + os.sep + model_name, monitor='val_loss', mode='min', verbose=1,
    #                                  save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=network_settings['patience'],
                                   verbose=1,
                                   restore_best_weights=True)

    hist = model.fit(x_t, y_t, epochs=network_settings['epochs'], validation_split=network_settings['validation_split'],
                     batch_size=network_settings['batch_size'], callbacks=[early_stopping], shuffle=False)

    ax_mse.plot(hist.history['val_loss'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mae.plot(hist.history['val_mae'], 'o--', label='Validation, fold: ' + str(fold))

    # plot history
    fig = plt.figure(
        figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction, fold: ' + str(fold))
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # # Prediction
    print("Predicting on inner-test data: \n")
    y_p = model.predict(x_tt, verbose=True)[:, 0]
    stats = utils_functions.get_stats(error=y_tt - y_p, name='', df_stats=pd.DataFrame(), gt=y_tt, est=y_p)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p, 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction, fold: ' + str(fold))
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model, stats, y_p, y_tt, hist


def arc1D_jeremy(input_shape, x_t, y_t, x_tt, y_tt, fold, ax_mse, ax_mae):
    # Define CNN parameters:

    network_settings = running_settings.network_settings

    input_acc = Input(input_shape)                                                  # shape: (300,6)
    dense = Dense(32, activation='relu', kernel_initializer='uniform')(input_acc)   # shape: Nx300x32
    dense = Dense(32, activation='relu', kernel_initializer='uniform')(dense)       # shape: Nx300x32
    flatten_final = Flatten()(dense)                                                # shape: Nx(300*32)
    output = Dense(1, activation='linear', kernel_initializer='uniform')(flatten_final)  # shape: Nx1

    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # reproducibility
    seed = 7
    np.random.seed(seed)

    # fit model
    # model_path = running_settings.settings['model_path']
    # model_name = running_settings.settings['model_name_fit'] + '_' + str(fold) + '.h5'
    # mdl_checkpoint = ModelCheckpoint(model_path + os.sep + model_name, monitor='val_loss', mode='min', verbose=1,
    #                                  save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0, patience=network_settings['patience'],
                                   verbose=1,
                                   restore_best_weights=True)

    hist = model.fit(x_t, y_t, epochs=network_settings['epochs'], validation_split=network_settings['validation_split'],
                     batch_size=network_settings['batch_size'], callbacks=[early_stopping], shuffle=False)

    ax_mse.plot(hist.history['val_loss'], 'o--', label='Validation, fold: ' + str(fold))
    ax_mae.plot(hist.history['val_mae'], 'o--', label='Validation, fold: ' + str(fold))

    # plot history
    fig = plt.figure(
        figsize=(running_settings.plot_settings['fig_width'], running_settings.plot_settings['fig_height']))
    ax = fig.add_subplot(221)
    ax.plot(hist.history['loss'], label='train')
    ax.plot(hist.history['val_loss'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('MSE, samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(222)
    ax.plot(hist.history['mae'], label='train')
    ax.plot(hist.history['val_mae'], label='validation')
    ax.set_xlabel('Epoch')
    ax.set_title('MAE, samples prediction, fold: ' + str(fold))
    ax.set_ylabel('MAE')
    ax.legend()
    plt.show(block=False)

    # # Prediction
    print("Predicting on inner-test data: \n")
    y_p = model.predict(x_tt, verbose=True)[:, 0]
    stats = utils_functions.get_stats(error=y_tt - y_p, name='', df_stats=pd.DataFrame(), gt=y_tt, est=y_p)

    # plotting prediction vs real data
    ax = fig.add_subplot(223)
    ax.plot(y_tt, 'o--', label='Ground Truth')
    ax.plot(y_p, 'o--', label='prediction on TEST data')
    ax.set_xlabel('TEST samples')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Samples prediction, fold: ' + str(fold))
    ax.legend()
    plt.show(block=False)

    ax = fig.add_subplot(224)
    ax.plot(abs(y_p - y_tt), 'o--', label='Absolute error')
    ax.set_xlabel('TEST samples')
    ax.set_title('samples prediction, fold: ' + str(fold))
    ax.set_ylabel('ABS(Ground Truth - Prediction)')
    ax.legend()
    plt.show(block=False)

    return model, stats, y_p, y_tt, hist

def create_model(input_shape):

    input_acc = Input(input_shape)                                          # (300,6)
    conv1 = Conv1D(64, 32, activation="relu", padding='same')(input_acc)    # shape: Nx300x64
    batch1 = BatchNormalization()(conv1)                                    # shape: Nx300x64
    max1 = MaxPooling1D(pool_size=3, padding='same')(batch1)                # shape: Nx100x64
    drop1 = Dropout(0.1)(max1)                                              # shape: Nx100x64
    conv2 = Conv1D(32, 16, activation="relu", padding='same')(drop1)        # shape: Nx100x32
    batch2 = BatchNormalization()(conv2)                                    # shape: Nx100x32
    max2 = MaxPooling1D(pool_size=3, padding='same')(batch2)                # shape: Nx34x32
    drop2 = Dropout(0.1)(max2)                                              # shape: Nx34x32
    conv3 = Conv1D(8, 3, activation="relu", padding='same')(drop2)          # shape Nx34x8
    batch3 = BatchNormalization()(conv3)                                    # shape Nx34x8
    max3 = MaxPooling1D(pool_size=3, padding='same')(batch3)                # shape Nx12x8
    flatten_final = Flatten()(max3)                                         # shape Nx96
    output = Dense(1, activation='relu')(flatten_final)  # shape Nx1

    model = Model(inputs=input_acc, outputs=output)
    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    return model

"""