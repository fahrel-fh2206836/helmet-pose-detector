// lib/screens/main_screen.dart
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

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  CameraController? _cam;
  HelmetPose? _pose;
  HelmetVideoClassifier? _streamer;

  StreamSubscription<({String label, double prob})>? _modelSub;
  String _status = 'not_tracking';
  bool _isStreamerRunning = true;
  ({String label, double prob})? _lastOutput;

  final _speed = SpeedService(
    windowSize: 5,
    emaAlpha: 0.35,
    maxAccuracyMeters: 25.0,
    // You can tweak locationSettings here if you want:
    // locationSettings: const LocationSettings(
    //   accuracy: LocationAccuracy.bestForNavigation,
    //   distanceFilter: 2,
    // ),
  );
  StreamSubscription<double>? _speedSub;
  StreamSubscription<SpeedSource>? _srcSub;
  StreamSubscription<SpeedStatus>? _statSub;

  double _kmhCurrent = 0.0;
  SpeedSource _kmhSource = SpeedSource.unknown;
  SpeedStatus _speedStatus = SpeedStatus.stopped;

  final _noti = NotiService();

  static const double _probThreshold = 0.7; // model confidence for "looking"
  static const double _minSpeedForAlert = 5.0;
  // static const Duration _cooldown = Duration(seconds: 10);

  DateTime? _lookingSince; // when we first saw a qualifying "looking" state
  // DateTime? _lastAlertAt; // last time we sent a warning
  Duration _requiredHold = const Duration(seconds: 6); // updated dynamically

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

  void _permissionCheck() async {
    hasCameraPermission = await PermissionService.hasCameraPermission();
    hasNotiPermission = await PermissionService.hasNotificationPermission();
    hasLocationPermission = await PermissionService.hasLocationPermission();
    if (mounted) {
      setState(() {});
    }
  }

  Duration _holdTimeForSpeed(double kmh) {
    // Your mapping:
    // >= 50 km/h -> 2s, >=25 -> 4s, >=15 -> 6s, else default (e.g., 8s)
    if (kmh >= 50) return const Duration(seconds: 2);
    if (kmh >= 25) return const Duration(seconds: 4);
    if (kmh >= 15) return const Duration(seconds: 6);
    return const Duration(seconds: 8); // between 5 and 14.9.
  }

  Future<void> _bootstrap() async {
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
      final cams = await availableCameras();
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );

      _cam = CameraController(
        front,
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _cam!.initialize();

      // Load your model (keeps helmet_pose.dart unchanged)
      _pose = await HelmetPose.load();

      // Wire the video streamer that feeds predict(Uint8List)
      _streamer = HelmetVideoClassifier(
        camera: _cam!,
        model: _pose!,
        maxFps: 8, // tune per device
      );

      _streamer?.start();

      // Subscribe to predictions: labels are "looking" / "not_looking"
      _modelSub = _streamer!.stream.listen((res) async {
        if (!mounted) return;

        // 1) Always update UI first so it reflects the latest model output
        setState(() {
          _lastOutput = res; // shows label + prob in your UI
          _status = 'tracking';
        });

        // 2) Now do alert logic
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
          // final inCooldown =
          //     _lastAlertAt != null && now.difference(_lastAlertAt!) < _cooldown;

          if ( /*!inCooldown &&*/ heldFor >= _requiredHold) {
            await _noti.showNotification(
              title: 'Warning!',
              body:
                  'Detected: looking at phone at ${speed.toStringAsFixed(1)} km/h',
            );
            // _lastAlertAt = now;
            _lookingSince = null; // require fresh hold after cooldown
          }
        } else {
          _lookingSince = null; // broke the "looking" streak
        }
      });
    } catch (e) {
      setState(() => _status = 'Error: $e');
    }
  }

  // Handle app lifecycle so camera stream pauses/resumes cleanly
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final cam = _cam;
    if (cam == null || !cam.value.isInitialized) return;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _speed.stop();
      _streamer?.stop();
      cam.pausePreview();
    } else if (state == AppLifecycleState.resumed) {
      _speed.start();
      cam.resumePreview();
      controlStreamer();
    }
  }

  void controlStreamer() {
    if (_isStreamerRunning) {
      _streamer?.start();
      setState(() {
        _status = "tracking";
      });
    } else {
      _streamer?.stop();
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
    _cam?.dispose();
    _pose?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _cam;
    if (cam == null || !cam.value.isInitialized) {
      return const Scaffold(
        body: Column(
          children: [
            CircularProgressIndicator(),
            Text(
              "Ensure that camera permission is granted. Otherwise, grant permission and restart the app.",
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
            const SizedBox(height: 20),
            AspectRatio(
              aspectRatio: cam.value.aspectRatio,
              child: CameraPreview(cam),
            ),
            const SizedBox(height: 20),
            IconWithText(
              iconData: Icons.phone_android,
              text:
                  "Looking at Phone: ${_lastOutput?.label == "looking" ? "Yes" : "No"}",
            ),
            const SizedBox(height: 20),
            IconWithText(
              iconData: Icons.motorcycle,
              text: 'Current Speed: ${_kmhCurrent.toStringAsFixed(2)} km/h',
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              style: ElevatedButton.styleFrom(
                backgroundColor: _isStreamerRunning
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
                _isStreamerRunning = !_isStreamerRunning;
                controlStreamer();
              },
              icon: Icon(
                !_isStreamerRunning ? Icons.power_settings_new : Icons.pause,
              ),
              label: Text(
                _isStreamerRunning ? 'Deactivate AI' : 'Activate AI',
                style: TextStyle(fontSize: 18),
                textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 20),
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
      ),
    );
  }

  Widget _buildModelData() {
    return Column(
      spacing: 8,
      children: [
        Text(
          'Output: ${_lastOutput != null ? _lastOutput!.label : "-"} (${_lastOutput != null ? (_lastOutput!.prob * 100).toStringAsFixed(1) : "-"}%)',
        ),
        Text('Status: $_status'),
      ],
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

  Widget _buildSpeedData() {
    return Column(
      children: [
        Text('Speed: ${_kmhCurrent.toStringAsFixed(1)} km/h'),
        Text('Source: ${prettySource(_kmhSource)}'),
        Text('GPS: ${prettyStatus(_speedStatus)}'),
      ],
    );
  }
}
