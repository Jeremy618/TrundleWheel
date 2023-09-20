# Walked distance dataset

This repository contains data of walking activities gathered by sensors embedded in commercial mobile phones
together with a meter-by-meter walked distance reference gathered through a sensorised trundle wheel.

Repository structure:

- Data: contains sensors data from walk activities
- **ModelFinalVersion**: contains used data and code for training and test the distance estimation model. 
  - FinalCNN.py is the main script
  - all_datas_2.csv is the CSV file containing inertial measurements (not GPS! Data with GPS information is in: Data/User1_Jeremy_Bezancon/all_datas.csv)
  - Inside package **settings**: 
    - running_settings.py contains multiple choices and flags to consider when running the code
    - **utils_functions.py**, **utils_cnn_architecture.py** and **utils_plots.py** contain functions called from FinalCNN.py 
- RedSwitchBLE contains code for the sensorised trundle wheel. The code uses the Arduino IDE and runs on a ESP32. It allows to count the distance walked by the person by detecting the presence of a magnet on one of the spokes of the trundle wheel. The distance (1 meter per detection) is sent via Bluetooth Low Energy in a custom service and characteristic.
- TrundleWheelApp contains the mobile phone app code to gather the sensors data, including the reference data from the trundle wheel. The app uses the Apache cordova framework.
