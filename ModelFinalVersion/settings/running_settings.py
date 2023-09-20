"""

Running settings file,
read settings and find usages so to run files as desired.
Change MOTHERPATH accordingly,

"""
import os

MOTHERPATH = r"C:\Users\ao4518\Desktop\PHD\TRUNDLE_wheel"

settings = {
    'chunk_test': 21600,  # SAMPLES = 60*60*6 = 6 minutes
    'chunk_test_time': 360000, # DURATION
    'chunk': 300,  # = 60*5 = 5 seconds
    'col_s': 4,  # first column
    'col_f': 10,  # last column
    'primi6min': True,
    'fit_model': False,
    'resampling_data': False,
    'normalize_data': False, # or 'minmax'
    'stats_save': False,
    'plot_save': False,
    'bst_model_save': True,
    'model_path': r"C:\Users\ao4518\Desktop\PHD\TRUNDLE_wheel\models",
    'model_images_path': MOTHERPATH + os.sep + "models_images",
    'bestparam_path': MOTHERPATH + os.sep + "models" + os.sep + "bestparam",
    'model_name_fit': 'mdl_arc1Dsim_nonorm_150_32',
    'model_name_load': 'mdl_arc1Dsim_nonorm_150_32_3',
    'bestparam_name': 'prm_arc1Dsim',
}

network_settings = {
    'epochs': 150,
    'batch_size': 32,
    'validation_split': 0.2,
    'patience': 25,
    'verbose': 1,
    'loss': 'mse',
    'metrics': ['mae', 'mape']
}

plot_settings = {
    'font_size': 18,
    'fig_width': 20,
    'fig_height': 10,
}
