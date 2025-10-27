// App's main/home screen

import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/enum/common_enums.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:helmet_detector_app/services/helmet_video_classifier.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
import 'package:helmet_detector_app/services/permission_service.dart';
import 'package:helmet_detector_app/services/speed_services.dart';
import 'package:helmet_detector_app/widgets/icon_with_text.dart';
import 'package:helmet_detector_app/widgets/live_camera_preview.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  // Object that controls camera configurations
  CameraController? _camController;

  // TFLite model wrapper and streaming classifier
  HelmetPose? _model;
  HelmetVideoClassifier? _streamer;
  StreamSubscription<({String label, double prob})>? _modelSub;

  // UI status (e.g., "tracking") and last model output
  String _status = 'not_tracking';
  bool _isServiceRunning = true;
  ({String label, double prob})? _lastOutput;

  // Smoothed speed pipeline
  final _speed = SpeedService(
    windowSize: 5,
    emaAlpha: 0.7,
    maxAccuracyMeters: 25.0,
  );
  StreamSubscription<double>? _speedSub;
  StreamSubscription<SpeedSource>? _srcSub;
  StreamSubscription<SpeedStatus>? _statSub;

  // Latest speed + meta for debug panel
  double _kmhCurrent = 0.0;
  SpeedSource _kmhSource = SpeedSource.unknown;
  SpeedStatus _speedStatus = SpeedStatus.stopped;

  // Local notification service
  final _noti = NotiService();

  // Detection policy thresholds (model confidence, ignore slow speeds, cooldown between alerts).
  static const double _probThreshold = 0.55;
  static const double _minSpeedForAlert = 5.0;
  static const Duration _cooldown = Duration(seconds: 10);

  DateTime? _lookingSince; // when we first saw a qualifying "looking" state
  DateTime? _lastAlertAt; // last time we sent a warning
  Duration _requiredHold = const Duration(
    seconds: 6,
  ); // dynamic time based on speed

  // Permission flags
  bool hasCameraPermission = false;
  bool hasNotiPermission = false;
  bool hasLocationPermission = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _permissionCheck();
    _bootstrap();
  }

  // read & show current permission states
  void _permissionCheck() async {
    hasCameraPermission = await PermissionService.hasCameraPermission();
    hasNotiPermission = await PermissionService.hasNotificationPermission();
    hasLocationPermission = await PermissionService.hasLocationPermission();
    if (mounted) {
      setState(() {});
    }
  }

  // Map current speed to how long "looking" must be held before an alert
  Duration _holdTimeForSpeed(double kmh) {
    if (kmh >= 50) return const Duration(seconds: 2);
    if (kmh >= 25) return const Duration(seconds: 4);
    if (kmh >= 15) return const Duration(seconds: 6);
    return const Duration(seconds: 8); // between 5 and <15
  }

  // Starts speed + camera + model + subscriptions
  Future<void> _bootstrap() async {
    // Starts speed tracking service and listeners to streams
    _speed.start();
    _speedSub = _speed.speedStream.listen((v) {
      if (!mounted) return;
      setState(() => _kmhCurrent = v);
    });

    _srcSub = _speed.sourceStream.listen((s) {
      if (!mounted) return;
      setState(() => _kmhSource = s);
    });

    _statSub = _speed.statusStream.listen((s) {
      if (!mounted) return;
      setState(() => _speedStatus = s);
    });

    try {
      // Pick a front camera (fallback to first available)
      final cams = await availableCameras();
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );

      // Configure camera (YUV → your isolate pipeline)
      _camController = CameraController(
        front,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _camController!.initialize();

      // Load your model (Default: assets/helmet_pose_fp16.tflite)
      _model = await HelmetPose.load(
        /*assetPath: 'assets/helmet_pose_int8_float32_io.tflite'*/
      );

      // Streamer: pulls frames, preprocesses on isolate, runs model
      _streamer = HelmetVideoClassifier(
        camera: _camController!,
        model: _model!,
        maxFps: 8, // tune per device
      );

      _streamer?.start();

      // Subscribe to (label, prob)
      _modelSub = _streamer!.stream.listen((res) async {
        if (!mounted) return;

        // 1) Always update UI first so it reflects the latest model output
        setState(() {
          _lastOutput = res; // shows label + prob in your UI
          _status = 'tracking';
        });

        // 2) Alert policy
        final now = DateTime.now();
        final speed = _kmhCurrent; // from SpeedService
        _requiredHold = _holdTimeForSpeed(speed);

        final bool isLooking =
            (res.label == 'looking') && (res.prob >= _probThreshold);

        // Skip alerts entirely when speed < 5 km/h (but UI already updated above)
        if (speed < _minSpeedForAlert) {
          _lookingSince = null; // reset any hold
          return;
        }

        if (isLooking) {
          _lookingSince ??= now;
          final heldFor = now.difference(_lookingSince!);
          final inCooldown =
              _lastAlertAt != null && now.difference(_lastAlertAt!) < _cooldown;

          // Only notify if held long enough and not in cooldown
          if (!inCooldown && heldFor >= _requiredHold) {
            unawaited(
              _noti.showNotification(
                title: 'Warning!',
                body:
                    'Detected: looking at phone at ${speed.toStringAsFixed(1)} km/h',
              )..catchError((e, st) {
                debugPrint('⚠️ Notification failed: $e');
                debugPrintStack(stackTrace: st);
              }),
            );
            _lastAlertAt = now;
            _lookingSince = null; // require fresh hold after cooldown
          }
        } else {
          _lookingSince = null; // break the "looking" streak
        }
      });
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  // Handle app lifecycle so camera stream pauses/resumes cleanly.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final cam = _camController;
    if (cam == null || !cam.value.isInitialized) return;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _speed.stop();
      _streamer?.stop();
      cam.pausePreview();
    } else if (state == AppLifecycleState.resumed) {
      _speed.start();
      cam.resumePreview();
      controlServices();
    }
  }

  // Controls services based on state
  void controlServices() {
    if (_isServiceRunning) {
      _streamer?.start();
      _speed.start();
      setState(() {
        _status = "tracking";
      });
    } else {
      _streamer?.stop();
      _speed.stop();
      setState(() {
        _status = "not_tracking";
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _speedSub?.cancel();
    _speed.dispose();
    _srcSub?.cancel();
    _statSub?.cancel();
    _modelSub?.cancel();
    _streamer?.dispose();
    _camController?.dispose();
    _model?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _camController;
    if (cam == null || !cam.value.isInitialized) {
      return const Scaffold(
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 20),
            Align(
              alignment: Alignment.center,
              child: Text(
                "Taking too long?\nEnsure that camera permission is granted.\nOtherwise, grant permission and restart the app.",
              ),
            ),
          ],
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'RideSafe',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        actions: [
          Padding(padding: const EdgeInsets.all(8.0), child: Text("ver. 1.0")),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            _buildPermissionText(),
            const SizedBox(height: 15),
            SizedBox(
              height: MediaQuery.of(context).size.height * .5,
              width: MediaQuery.of(context).size.width * 1,
              child: LiveCameraView(controller: _camController!),
            ),
            const SizedBox(height: 15),
            IconWithText(
              iconData: Icons.phone_android,
              text:
                  "Looking at Phone: ${_lastOutput?.label == "looking" ? "Yes" : "No"}",
            ),
            const SizedBox(height: 15),
            IconWithText(
              iconData: Icons.motorcycle,
              text: 'Current Speed: ${_kmhCurrent.toStringAsFixed(2)} km/h',
            ),
            const SizedBox(height: 15),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: _isServiceRunning
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
                _isServiceRunning = !_isServiceRunning;
                controlServices();
              },
              icon: Icon(
                !_isServiceRunning ? Icons.power_settings_new : Icons.pause,
              ),
              label: Text(
                _isServiceRunning
                    ? 'Deactivate Detection'
                    : 'Activate Detection',
                style: TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 15),
            ExpansionTile(
              leading: Icon(Icons.info, color: Colors.green),
              title: Text('Technical Data'),
              subtitle: Text('Tap to expand debugs'),
              children: [
                Text("Model Result & Status"),
                Container(
                  padding: EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.green),
                  ),
                  child: _buildModelData(),
                ),
                const SizedBox(height: 8),
                Text("Speed Data"),
                Container(
                  padding: EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.green),
                  ),
                  child: _buildSpeedData(),
                ),
                const SizedBox(height: 8),
              ],
            ),
            const SizedBox(height: 8),
          ],
        ),
      ),
    );
  }

  Widget _buildPermissionText() {
    return Column(
      children: [
        Text(
          'Please allow all the permissions that has been asked in order for the app to function properly:',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 16),
        ),
        SizedBox(height: 5),
        Text("Location: ${hasLocationPermission ? "Yes" : "No"}"),
        Text("Notification: ${hasNotiPermission ? "Yes" : "No"}"),
        Text("Camera: ${hasCameraPermission ? "Yes" : "No"}"),
      ],
    );
  }

  Widget _buildModelData() {
    return Column(
      spacing: 8,
      children: [
        Text(
          'Last Output: ${_lastOutput != null ? _lastOutput!.label : "-"} (${_lastOutput != null ? _lastOutput!.prob * 100 : "-"}%)',
        ),
        Text('Status: $_status'),
      ],
    );
  }

  Widget _buildSpeedData() {
    return Column(
      children: [
        Text("Speed: $_kmhCurrent km/h"),
        Text('Source: ${prettySource(_kmhSource)}'),
        Text('Status: ${prettyStatus(_speedStatus)}'),
      ],
    );
  }
}
