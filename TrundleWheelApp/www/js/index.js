/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import files from './files.js'

// Wait for the deviceready event before using any of Cordova's device APIs.
// See https://cordova.apache.org/docs/en/latest/cordova/events/events.html#deviceready
document.addEventListener('deviceready', onDeviceReady, false);

const deviceId = '24:6F:28:7B:DE:A2'; // ID of the target BLE device
const serviceUuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'; // UUID of the BLE service
const characteristicUuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'; // UUID of the BLE characteristic

const TMP_FILENAME = 'walkDatas.txt';
let logger;

function onDeviceReady() {
    // Cordova is now initialized. Have fun!

    console.log('Running cordova-' + cordova.platformId + '@' + cordova.version);
    document.getElementById('deviceready').classList.add('ready');

    console.log(cordova.file);

    var distance = 0;
    var dataContainer = document.getElementById('dataContainer');
    dataContainer.textContent = distance;
    var isConnected = false; // True if the phone is connected to the ESP32
    var connectionInfo = document.getElementById('connectionInfo');
    connectionInfo.textContent = "not connected to ESP32";
    var isWalking = false;

    //__________________________________________buttons__________________________________
    var startButton = document.getElementById('startButton'); // start_button
    startButton.addEventListener('click', function() {
        console.log("startButton");
        if (isWalking == false){
            console.log("start to walk");
            distance = 0;
            dataContainer.textContent = distance;
            isWalking = true;
            start();
        }
    });
    var endButton = document.getElementById('endButton'); // end_button
    endButton.addEventListener('click', function() {
        console.log("endButton");
        if(isWalking == true){
            isWalking = false;
            end();
            console.log("walk finished");
        }
    });

    var connectButton = document.getElementById('connectButton'); // connect_button
    connectButton.addEventListener('click', async function() {
        console.log("connectButton");
        if(!isConnected) ESP32Connection();
    });

}

// Try to connect to the ESP32 device
function ESP32Connection(){
    console.log('Connecting to ESP32');
    connectionInfo.textContent = 'Connecting to ESP32';
    ble.connect(deviceId, function(device) {
        console.log('Connected to device: ' + JSON.stringify(device));
        connectionInfo.textContent = 'Connected to ESP32';
        ESP32Notification();
        isConnected = true;
    }, function(error) {
        console.log('BLE connection error: ' + error);
        connectionInfo.textContent = 'Not connected to ESP32';
        isConnected = false;
    });
}

// Start receiving notifications from a BLE characteristic
function ESP32Notification(){
    ble.startNotification(
        deviceId, // ID of the BLE device
        serviceUuid, // UUID of the BLE service
        characteristicUuid, // UUID of the BLE characteristic
        function(data) {
            var dataArray = new Uint8Array(data);
            distance = (dataArray[3] << 24) | (dataArray[2] << 16) | (dataArray[1] << 8) | dataArray[0];
            console.log('distance= ' + distance);
            dataContainer.textContent = distance;
        },
        function(error) {
            console.log('BLE notification error: ' + error);
        }
    );
}

async function start(){
    try {
        logger = await files.createLog(TMP_FILENAME);
        let filePath = await files.getFilePath(TMP_FILENAME);
        console.log(filePath);
        // save file in the path: "Ce PC\Redmi Note 9 Pro\Espace de stockage interne partagé\Android\data\io.cordova.hellocordova\files"
    } catch (e) {
        console.error(e);
    }
    logger.log('start');
}

async function end(){
    try {
        logger = await files.createLog(TMP_FILENAME);
    } catch (e) {
        console.error(e);
    }
    logger.log('end');
}
