// lib/screens/main_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:helmet_detector_app/services/helmet_stream_service.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
import 'package:helmet_detector_app/services/permission_service.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});
  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  CameraController? _cam;
  HelmetPose? _pose;
  HelmetVideoClassifier? _streamer;

  final _noti = NotiService();
  StreamSubscription<({String label, double prob})>? _sub;

  String _status = 'Initializing…';
  ({String label, double prob})? _last;

  static const double alertThreshold = 0.80; // for "looking"

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    try {
      await PermissionService.requestAllPermissions();
      await _noti.initNotification();

      // Select back camera
      final cams = await availableCameras();
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );

      _cam = CameraController(
        front,
        ResolutionPreset.medium,
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
        maxFps: 30, // tune per device
      );

      // Subscribe to predictions: labels are "looking" / "not_looking"
      _sub = _streamer!.stream.listen((res) async {
        if (!mounted) return;
        setState(() => _last = res);

        // Optional: notify only when "looking" is confident
        if (res.label == 'looking' && res.prob >= alertThreshold) {
          await _noti.showNotification(
            id: 1,
            title: 'Unsafe behaviour',
            body: 'Detected: looking at phone',
          );
        }
      });

      await _streamer!.start();
      setState(() => _status = 'Running');
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
      _streamer?.stop();
      cam.pausePreview();
    } else if (state == AppLifecycleState.resumed) {
      cam.resumePreview();
      _streamer?.start();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _sub?.cancel();
    _streamer?.dispose();
    _cam?.dispose();
    _pose?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _cam;
    if (cam == null || !cam.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Looking / Not Looking — Stream')),
      body: Column(
        children: [
          AspectRatio(
            aspectRatio: cam.value.aspectRatio,
            child: CameraPreview(cam),
          ),
          const SizedBox(height: 8),
          if (_last != null)
            Text(
              '${_last!.label}  (${(_last!.prob * 100).toStringAsFixed(1)}%)',
              style: const TextStyle(fontSize: 16),
            )
          else
            Text(_status),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton(
                onPressed: () => _streamer?.start(),
                child: const Text('Start'),
              ),
              const SizedBox(width: 12),
              OutlinedButton(
                onPressed: () => _streamer?.stop(),
                child: const Text('Stop'),
              ),
            ],
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
