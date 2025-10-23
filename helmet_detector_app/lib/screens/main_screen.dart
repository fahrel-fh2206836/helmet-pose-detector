// lib/screens/main_screen.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:helmet_detector_app/services/helmet_stream_service.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
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

  final _noti = NotiService();
  StreamSubscription<({String label, double prob})>? _sub;

  String _status = 'not_tracking';
  bool _isStreamerRunning = false;
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

      // Subscribe to predictions: labels are "looking" / "not_looking"
      _sub = _streamer!.stream.listen((res) async {
        if (!mounted) return;
        setState(() => _last = res);

        // Optional: notify only when "looking" is confident
        if (res.label == 'looking' && res.prob >= alertThreshold) {
          await _noti.showNotification(
            title: 'Warning!',
            body: 'Detected: looking at phone!',
          );
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
      _streamer?.stop();
      cam.pausePreview();
    } else if (state == AppLifecycleState.resumed) {
      cam.resumePreview();
      controlStreamer();
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

  void controlStreamer() {
    _isStreamerRunning = !_isStreamerRunning;
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
  Widget build(BuildContext context) {
    final cam = _cam;
    if (cam == null || !cam.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'RideSafe',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          AspectRatio(
            aspectRatio: cam.value.aspectRatio,
            child: CameraPreview(cam),
          ),
          const SizedBox(height: 20),
          IconWithText(
            iconData: Icons.phone_android,
            text:
                "Looking at Phone: ${_last?.label == "looking" ? "Yes" : "No"}",
          ),
          const SizedBox(height: 20),
          IconWithText(
            iconData: Icons.motorcycle,
            text: /*'Current Speed: ${_currentSpeed.toStringAsFixed(2)} km/h'*/
                "",
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: _isStreamerRunning
                  ? const Color(0xFF12EB66)
                  : Colors.grey,
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(30),
              ),
            ),
            onPressed: () {
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
            height: 80,
            width: 250,
            padding: EdgeInsets.all(10),
            decoration: BoxDecoration(border: Border.all(color: Colors.green)),
            child: _buildModelData(),
          ),
        ],
      ),
    );
  }

  Widget _buildModelData() {
    return Column(
      children: [
        Text(
          'Output: ${_last!.label} (${(_last!.prob * 100).toStringAsFixed(1)}%)',
          style: const TextStyle(fontSize: 16),
        ),
        const SizedBox(height: 8),
        Text('Status: $_status', style: const TextStyle(fontSize: 16)),
      ],
    );
  }
}
