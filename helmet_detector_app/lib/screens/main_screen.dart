import 'dart:async';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/utils/image_utils.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  bool _isActivated = false;
  Timer? _activationTimer;
  int _secondsElapsed = 0;
  double _currentSpeed = 0.0;

  // AI and Camera
  CameraController? _cameraController;
  Interpreter? _helmetInterpreter;
  Interpreter? _classifierInterpreter;
  bool _helmetDetected = false;
  bool _isLookingAtPhone = false;

  @override
  void initState() {
    super.initState();
    _startTracking();
    _loadModels();
  }

  Future<void> _loadModels() async {
    _helmetInterpreter = await Interpreter.fromAsset(
      'assets/best_float32.tflite',
    );
    _classifierInterpreter = await Interpreter.fromAsset(
      'assets/classifier_model.tflite',
    );
  }

  Future<void> _startTracking() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return;
    }
    if (permission == LocationPermission.deniedForever) return;

    Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 5,
      ),
    ).listen((Position position) {
      setState(() {
        _currentSpeed = (position.speed) * 3.6;
      });
    });
  }

  void _startTimer() {
    _secondsElapsed = 0;
    _activationTimer?.cancel();
    _activationTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _secondsElapsed++;
      });
    });
  }

  void _stopTimer() {
    _activationTimer?.cancel();
    _secondsElapsed = 0;
  }

  // Future<void> _runPipeline(CameraImage image) async {
  //   final inputImage = await convertCameraImage(image);

  //   final inputTensor = inputImage.tensorBuffer.buffer;
  //   final outputBuffer = TensorBuffer.createFixedSize([
  //     1,
  //     2,
  //   ], TfLiteType.float32);
  //   _helmetInterpreter!.run(inputTensor, outputBuffer.buffer);
  //   final helmetProb = outputBuffer.getDoubleList()[0];

  //   setState(() {
  //     _helmetDetected = helmetProb > 0.5;
  //   });

  //   if (_helmetDetected) {
  //     final output2 = TensorBuffer.createFixedSize([1, 2], TfLiteType.float32);
  //     _classifierInterpreter!.run(inputTensor, output2.buffer);
  //     final phoneProb = output2.getDoubleList()[0];

  //     setState(() {
  //       _isLookingAtPhone = phoneProb > 0.5;
  //     });
  //   } else {
  //     setState(() {
  //       _isLookingAtPhone = false;
  //     });
  //   }
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16.0),
              child: Text(
                'DriveSafe',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
            ),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Stay Focused, Drive Safe',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Colors.deepOrange,
                ),
              ),
            ),
            const SizedBox(height: 8),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Please allow all the permissions that has been asked in order for the app to function properly',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: 300,
              height: 50,
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isActivated
                      ? const Color(0xFF12EB66)
                      : Colors.grey,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                onPressed: () {
                  setState(() {
                    _isActivated = !_isActivated;
                  });
                  if (_isActivated) {
                    _startTimer();
                    // _cameraController?.startImageStream(
                    //   (image) => _runPipeline(image),
                    // );
                  } else {
                    _stopTimer();
                    _cameraController?.stopImageStream();
                  }
                },
                icon: Icon(
                  !_isActivated ? Icons.power_settings_new : Icons.pause,
                ),
                label: Expanded(
                  child: Text(
                    _isActivated
                        ? 'Deactive AI & Camera'
                        : 'Activate AI & Camera',
                    style: TextStyle(fontSize: 18),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.timer, color: Colors.green),
                SizedBox(width: 20),
                Text(
                  _isActivated
                      ? 'Tracking for $_secondsElapsed seconds'
                      : 'Currently Not Tracking',
                  style: const TextStyle(fontSize: 16),
                ),
              ],
            ),
            const SizedBox(height: 30),
            _buildSpeedWidget(),
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.info), label: ''),
        ],
        onTap: (index) {},
      ),
    );
  }

  Widget _buildSpeedWidget() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.motorcycle, color: Colors.green),
        const SizedBox(width: 8),
        Text(
          'Current Speed: ${_currentSpeed.toStringAsFixed(2)} km/h',
          style: const TextStyle(fontSize: 16),
        ),
      ],
    );
  }
}
