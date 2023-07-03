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

// Wait for the deviceready event before using any of Cordova's device APIs.
// See https://cordova.apache.org/docs/en/latest/cordova/events/events.html#deviceready
document.addEventListener('deviceready', onDeviceReady, false);

function onDeviceReady() {
    // Cordova is now initialized. Have fun!

    console.log('Running cordova-' + cordova.platformId + '@' + cordova.version);
    document.getElementById('deviceready').classList.add('ready');

    var deviceId = '24:6F:28:7B:DE:A2'; // ID of the target BLE device
    var serviceUuid = '4fafc201-1fb5-459e-8fcc-c5c9c331914b'; // UUID of the BLE service
    var characteristicUuid = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'; // UUID of the BLE characteristic

    var distance = 0;
    // Update the content of the data container element with the received data
    var dataContainer = document.getElementById('dataContainer');
    dataContainer.textContent = distance;

    // Scan for BLE devices
    ble.scan([], 5, function(device) {
        console.log('Device found: ' + JSON.stringify(device));
        // Do something with the found device
    }, function(error) {
        console.log('BLE scan error: ' + error);
    });

    // Connect to a BLE device
    ble.connect(deviceId, function(device) {
        console.log('Connected to device: ' + JSON.stringify(device));
        var isConnected = document.getElementById('isConnected');
        isConnected.textContent = 'Connected to ESP32';
        // Start receiving notifications from a BLE characteristic
        ble.startNotification(
            deviceId, // ID of the BLE device
            serviceUuid, // UUID of the BLE service
            characteristicUuid, // UUID of the BLE characteristic
            function(data) {
                var dataArray = new Uint8Array(data);
                distance = (dataArray[3] << 24) | (dataArray[2] << 16) | (dataArray[1] << 8) | dataArray[0];
                console.log('distance= '+distance);
                dataContainer.textContent = distance;
            },
            function(error) {
                console.log('BLE notification error: ' + error);
            }
        );

    }, function(error) {
        console.log('BLE connection error: ' + error);
        isConnected.textContent = 'Not connected to ESP32';
    });


}
