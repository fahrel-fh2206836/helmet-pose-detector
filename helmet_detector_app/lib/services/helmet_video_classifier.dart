// lib/services/helmet_stream.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import '../models/helmet_pose.dart';

/// Adapts Camera stream frames to the HelmetPose.predict(Uint8List) contract.
class HelmetVideoClassifier {
  final CameraController camera;
  final HelmetPose model;

  /// Emits ({label, prob}) for each processed frame.
  final StreamController<({String label, double prob})> _controller =
      StreamController.broadcast();

  Stream<({String label, double prob})> get stream => _controller.stream;

  bool _busy = false;
  bool _started = false;

  /// process at most this many frames per second (simple throttle)
  final int maxFps;

  HelmetVideoClassifier({
    required this.camera,
    required this.model,
    this.maxFps = 8,
  });

  Future<void> start() async {
    if (_started) return;
    _started = true;

    // Make sure camera is initialized in YUV420
    if (!camera.value.isInitialized) {
      await camera.initialize();
    }
    if (camera.value.isStreamingImages) return;

    Duration minGap = Duration(milliseconds: (1000 / maxFps).floor());
    var last = DateTime.fromMillisecondsSinceEpoch(0);

    await camera.startImageStream((CameraImage image) async {
      final now = DateTime.now();
      if (_busy || (now.difference(last) < minGap)) return;
      _busy = true;
      last = now;

      try {
        final bytes = _cameraImageToJpeg(
          image,
          jpegQuality: 60,
        ); // YUV420 -> RGB -> JPEG
        final res = await model.predict(bytes); // uses your helmet_pose.dart
        if (!_controller.isClosed) _controller.add(res);
      } catch (_) {
        // swallow frame errors to keep stream alive
      } finally {
        _busy = false;
      }
    });
  }

  Future<void> stop() async {
    if (!_started) return;
    _started = false;
    if (camera.value.isStreamingImages) {
      await camera.stopImageStream();
    }
  }

  Future<void> dispose() async {
    await stop();
    await _controller.close();
  }
}

/// ---- Helpers ----
/// Convert YUV420 (CameraImage) to a compressed RGB image (JPEG) that
/// HelmetPose.predict() can decode. We keep quality modest for speed.
Uint8List _cameraImageToJpeg(CameraImage image, {int jpegQuality = 80}) {
  final w = image.width;
  final h = image.height;

  // Planes: Y, U, V
  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];

  final imgOut = img.Image(width: w, height: h);

  final uvRowStride = uPlane.bytesPerRow;
  final uvPixelStride = uPlane.bytesPerPixel ?? 1;

  // YUV420 -> RGB
  // BT.601-ish conversion, clamped to [0,255]
  int clamp(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);

  for (int y = 0; y < h; y++) {
    final int yRow = y * yPlane.bytesPerRow;
    final int uvRow = (y >> 1) * uvRowStride;

    for (int x = 0; x < w; x++) {
      final int yIndex = yRow + x;
      final int uvIndex = uvRow + (x >> 1) * uvPixelStride;

      final int Y = yPlane.bytes[yIndex];
      final int U = uPlane.bytes[uvIndex];
      final int V = vPlane.bytes[uvIndex];

      int r = Y + ((1436 * (V - 128)) >> 10);
      int g = Y - ((46549 * (U - 128)) >> 17) - ((93604 * (V - 128)) >> 17);
      int b = Y + ((1814 * (U - 128)) >> 10);

      imgOut.setPixelRgba(x, y, clamp(r), clamp(g), clamp(b), 255);
    }
  }

  // Compress to JPEG so HelmetPose.predict() can decode bytes.
  return Uint8List.fromList(img.encodeJpg(imgOut, quality: jpegQuality));
}
