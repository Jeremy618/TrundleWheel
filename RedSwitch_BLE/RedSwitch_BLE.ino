#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

#define SENSOR_PIN 27
#define LED_PIN 12
#define MIN_REVOLUTION_INTERVAL 500 // min time of a revolution of the wheel (if the person is running at 12 km/h) ------------- to test
// 500 ms -> 7.2 km/h
// run at 12 km/h = 3.3 m/s <=> 0.3 s/m
// if the person is running, the interval must be short to detect every revolution
// if the person is walking slowly, the interval must be long to not detect several times the same revolution
#define MAX_SHIPMENT_INTERVAL 5000 // max time between two shipments

BLEServer* pServer = NULL;
BLECharacteristic* pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;
unsigned long lastConnection = millis(); // last time the device was connected

int distance = 0; // total distance
int data_read = 0; // current entering the pin
float voltage = 0; // conversion of the current to volts
unsigned long lastDetection = 0; // last time the Magnet was detected
unsigned long lastShipment = 0; // last time a value was sent to the app
bool detected = false; // True if the magnet is detected, else -> False

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      Serial.println("device connected");
      deviceConnected = true;
    };

    void onDisconnect(BLEServer* pServer) {
      Serial.println("device disconnected");
      deviceConnected = false;
    }
};
class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic){ 
      std::string value = pCharacteristic->getValue();

      Serial.print("value.length = ");
      Serial.println(value.length());
      if (value.length() > 0) {
        Serial.println("*********");
        Serial.print("New value: ");
        distance = 0;
        int buffer = 0;
        int power = 1;
        for (int i = 1; i < value.length(); i++){
          power = power*10;
        }
        for (int i = 0; i < value.length(); i++){
          Serial.print(value[i]);
          buffer = value[i]-48;
          distance += buffer*power;
          power /= 10;
        }
        Serial.println();
        Serial.print("distance = ");
        Serial.println(distance);
        Serial.println("*********");
      }

    }
};

void setup() {
  Serial.begin(115200);

    // Create the BLE Device
  BLEDevice::init("ESP32");

  // Create the BLE Server
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // Create the BLE Service
  BLEService *pService = pServer->createService(SERVICE_UUID);

  // Create a BLE Characteristic
  pCharacteristic = pService->createCharacteristic( // Only Read and Notify are needed
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE  |
                      BLECharacteristic::PROPERTY_NOTIFY 
                    );

  pCharacteristic->setCallbacks(new MyCallbacks());
  // https://www.bluetooth.com/specifications/gatt/viewer?attributeXmlFile=org.bluetooth.descriptor.gatt.client_characteristic_configuration.xml
  // Create a BLE Descriptor
  pCharacteristic->addDescriptor(new BLE2902());

  // Start the service
  pService->start();

  // Start advertising
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);  // set value to 0x00 to not advertise this parameter
  BLEDevice::startAdvertising();
  Serial.println("Waiting a client connection to notify...");

  pinMode(SENSOR_PIN, INPUT);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  // if the tungelement sensor detects a magnet, it will let the current flow through him
  // digitalWrite(LED_PIN, LOW);

  // device is connected
  if (deviceConnected) {
    lastConnection == millis();
  }
  // device is disconnecting
  if (!deviceConnected && oldDeviceConnected) {
    unsigned long currentConnection = millis();
    if((currentConnection - lastConnection) > 500){// give the bluetooth stack the chance to get things ready
      pServer->startAdvertising(); // restart advertising
      Serial.println("start advertising");
      oldDeviceConnected = deviceConnected;
    }
  }
  // device is connecting
  if (deviceConnected && !oldDeviceConnected) {
      // do stuff here on connecting
      oldDeviceConnected = deviceConnected;
  }

  unsigned long currentDetection = millis();
  unsigned long currentShipment = millis();

  // read data sensor
  data_read = analogRead(SENSOR_PIN);
  voltage = data_read * (3.3 / 4095);

  if ((currentDetection - lastDetection) >= MIN_REVOLUTION_INTERVAL) // if the minimum time for a revolution has elapsed
  { 
    if (voltage >= 3.29) // if the magnet is in front of the sensor
    {
      if(detected == false) // if the magnet stay in front of the sensor it will detect it just one time
      {
        detected = true;
        distance++;
        // digitalWrite(LED_PIN, HIGH);
        Serial.println("Total distance = " + String(distance) + " m");
        // send data to the app
        if (deviceConnected) {
          pCharacteristic->setValue((uint8_t*)&distance, 4);
          pCharacteristic->notify();
        }
        lastDetection = currentDetection; // save the last time the Magnet was detected
        lastShipment = currentShipment; // save the last time a value was sent
      }    
    }else{
      detected = false;
    }
  }

  if ((currentShipment - lastShipment) >= MAX_SHIPMENT_INTERVAL){ // if the max time between two shipments has elapsed
    Serial.println("max time");
    // send data to the app
    if (deviceConnected) {
      pCharacteristic->setValue((uint8_t*)&distance, 4);
      pCharacteristic->notify();
    }
    lastShipment = currentShipment; // save the last time a value was sent
  } 
  
}
