import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/util/image_utils.dart';
import 'package:helmet_detector_app/util/noti_service.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

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

  CameraController? _cameraController;
  bool helmetDetected = false;
  bool isLooking = false;
  Interpreter? _helmetModel;
  Interpreter? _lookingModel;
  bool _cameraPermissionGranted = false;

  Timer? _lookingTimer;
  int _lookingSeconds = 0;

  @override
  void initState() {
    super.initState();
    _startTracking();
    _loadModels();
    _checkCameraPermissionAndInitialize();
  }

  Future<void> _checkCameraPermissionAndInitialize() async {
    final status = await Permission.camera.status;
    if (status.isGranted) {
      _cameraPermissionGranted = true;
      await _initializeCamera();
    } else if (status.isDenied || status.isRestricted) {
      final result = await Permission.camera.request();
      if (result.isGranted) {
        _cameraPermissionGranted = true;
        await _initializeCamera();
      } else {
        setState(() {
          _cameraPermissionGranted = false;
        });
      }
    } else if (status.isPermanentlyDenied) {
      await openAppSettings();
    }
  }

  Future<void> _loadModels() async {
    _helmetModel = await Interpreter.fromAsset('assets/yolov8s.tflite');
    // _lookingModel = await Interpreter.fromAsset('looking_cnn.tflite');
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
    );

    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController!.initialize();

    _cameraController!.startImageStream((CameraImage image) async {
      if (_isActivated) {
        // await _runModelPipeline(image);

        if (isLooking) {
          _lookingTimer ??= Timer.periodic(const Duration(seconds: 1), (
            timer,
          ) async {
            _lookingSeconds++;
            if (_lookingSeconds >= 5 && _currentSpeed > 1) {
              await NotiService().showNotification(
                title: 'Warning',
                body: 'You are looking at your phone!',
              );
              _lookingSeconds = 0;
              _lookingTimer?.cancel();
              _lookingTimer = null;
            }
          });
        } else {
          _lookingTimer?.cancel();
          _lookingTimer = null;
          _lookingSeconds = 0;
        }
      }
    });

    setState(() {});
  }

  // Future<void> _runModelPipeline(CameraImage image) async {
  //   final helmetResult = await runHelmetDetection(image);
  //   setState(() {
  //     helmetDetected = helmetResult;
  //   });

  //   // if (helmetDetected) {
  //   //   final lookResult = await runLookingClassification(image);
  //   //   setState(() {
  //   //     isLooking = lookResult;
  //   //   });
  //   // }
  // }

  // Future<bool> runHelmetDetection(CameraImage image) async {
  //   final Float32List input = await CameraImageUtils.preprocessCameraImage(
  //     image,
  //   );
  //   final outputTensor = _helmetModel!.getOutputTensor(0);
  //   final outputShape = outputTensor.shape; // [1, 6, 3549]
  //   final int valuesPerPrediction = outputShape[1];
  //   final int numPredictions = outputShape[2];
  //   final int outputLength = valuesPerPrediction * numPredictions;

  //   final outputBuffer = Float32List(outputLength);

  //   _helmetModel!.run(
  //     input.buffer.asUint8List(),
  //     outputBuffer.buffer.asUint8List(),
  //   );

  //   for (int i = 0; i < numPredictions; i++) {
  //     final double objectness = outputBuffer[4 * numPredictions + i];
  //     final double classScore = outputBuffer[5 * numPredictions + i];
  //     final double confidence = objectness * classScore;
  //     if (confidence > 0.5) {
  //       return true;
  //     }
  //   }

  //   return false;
  // }

  // Future<bool> runLookingClassification(CameraImage image) async {
  //   final Float32List input = await CameraImageUtils.preprocessCameraImage(
  //     image,
  //     targetSize: 300,
  //   );

  //   final outputTensor = _lookingModel!.getOutputTensor(0);
  //   final outputShape = outputTensor.shape;

  //   final outputBuffer = Float32List(outputShape.reduce((a, b) => a * b));
  //   _lookingModel!.run(
  //     input.buffer.asUint8List(),
  //     outputBuffer.buffer.asUint8List(),
  //   );

  //   final scores = outputBuffer;
  //   final maxScoreIndex = scores.indexOf(
  //     scores.reduce((a, b) => a > b ? a : b),
  //   );

  //   return maxScoreIndex == 1;
  // }

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
    _lookingTimer?.cancel();
    _lookingTimer = null;
    _lookingSeconds = 0;
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _helmetModel?.close();
    _lookingModel?.close();
    _lookingTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: SingleChildScrollView(
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
              if (_cameraController != null &&
                  _cameraController!.value.isInitialized)
                SizedBox(
                  width: 300,
                  height: 300,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: CameraPreview(_cameraController!),
                  ),
                )
              else if (!_cameraPermissionGranted)
                const SizedBox(
                  width: 300,
                  height: 300,
                  child: Center(child: Text('Camera permission is required')),
                )
              else
                const SizedBox(
                  width: 300,
                  height: 300,
                  child: Center(child: CircularProgressIndicator()),
                ),
              const SizedBox(height: 24),
              SizedBox(
                width: 200,
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

                      if (_isActivated) {
                        _startTimer();
                      } else {
                        _stopTimer();
                        helmetDetected = false;
                        isLooking = false;
                      }
                    });
                  },
                  icon: Icon(
                    !_isActivated ? Icons.power_settings_new : Icons.pause,
                  ),
                  label: Expanded(
                    child: Text(
                      _isActivated ? 'Deactivate AI' : 'Activate AI',
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
              const SizedBox(height: 20),
              Text(
                "Helmet Detected: ${helmetDetected ? "Yes" : "No"}",
                style: const TextStyle(fontSize: 18),
              ),
              const SizedBox(height: 8),
              Text(
                "Looking at Phone: ${isLooking ? "Yes" : "No"}",
                style: const TextStyle(fontSize: 18),
              ),
            ],
          ),
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
