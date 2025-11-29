// Service used to receive stream frames (images), preprocess and formatted into tensor input, ready for inferencing.
// Heavy Preprocessing and formatting done on an isolate thread (background) to keep the UI Thread smooth.

import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:helmet_detector_app/services/preprocessing_service.dart';

class VideoStreamerService {
  final CameraController camera;
  final HelmetPose model;

  // Broadcasts values of (label, prob) to listeners in the UI.
  final StreamController<({String label, double prob})> _controller =
      StreamController.broadcast();
  Stream<({String label, double prob})> get stream => _controller.stream;

  // Max inference rate (frames per second)
  final int maxFps;

  VideoStreamerService({
    required this.camera,
    required this.model,
    this.maxFps = 8,
  });

  bool _started =
      false; // Boolean whether service has started, prevents double starting.
  bool _busy =
      false; // Simple backpressure guard, prevent a frame from being processed if previous frame hasn't been fully processed.
  Timer? _ticker; // Periodic trigger to run inference
  CameraImage? _latest; // Newest camera frame (we keep only the latest)

  // Start stream service
  Future<void> start() async {
    if (_started) return;
    _started = true;

    // Ensure camera is ready
    if (!camera.value.isInitialized) {
      await camera.initialize();
    }

    /* 
    Begin receiving frames if not already. We only store the most recent frame
    to avoid backlog; inference is rate-limited by _ticker.
    */
    if (!camera.value.isStreamingImages) {
      await camera.startImageStream((CameraImage image) {
        _latest = image; // Store the newest frame
      });
    }

    // Create a periodic timer that attempts inference at most `maxFps`
    final periodMs = (1000 / maxFps).floor();
    _ticker = Timer.periodic(Duration(milliseconds: periodMs), (_) async {
      if (_busy) return; // if previous run still executing, skip
      final imgNow = _latest;
      if (imgNow == null) return; // nothing to process yet
      _busy = true;

      try {
        // Pack planes + metadata for isolate
        final message = packForIsolate(imgNow, model);
        // Do all heavy work off the UI isolate
        final res =
            await compute<Map<String, dynamic>, ({String label, double prob})>(
              preprocessIsolate,
              message,
            );

        // Emit the result to listeners if stream is still open
        if (!_controller.isClosed) _controller.add(res);
      } catch (_) {
        // Swallow errors to keep the stream alive
      } finally {
        _busy = false;
      }
    });
  }

  // Stops stream service
  Future<void> stop() async {
    if (!_started) return;
    _started = false;

    // Stop periodic trigger and clear last frame
    _ticker?.cancel();
    _ticker = null;
    _latest = null;

    // Stop camera stream if active
    if (camera.value.isStreamingImages) {
      await camera.stopImageStream();
    }
  }

  Future<void> dispose() async {
    await stop();
    await _controller.close();
  }
}
