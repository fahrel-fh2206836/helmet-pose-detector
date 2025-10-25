// lib/services/helmet_video_classifier.dart
import 'dart:async';
import 'dart:isolate';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart'; // compute()
import 'package:image/image.dart' as img;
import '../models/helmet_pose.dart';

class HelmetVideoClassifier {
  final CameraController camera;
  final HelmetPose model;

  final StreamController<({String label, double prob})> _controller =
      StreamController.broadcast();
  Stream<({String label, double prob})> get stream => _controller.stream;

  final int maxFps;

  HelmetVideoClassifier({
    required this.camera,
    required this.model,
    this.maxFps = 8,
  });

  bool _started = false;
  bool _busy = false;
  Timer? _ticker;
  CameraImage? _latest;
  bool _isNCHW = false; // read once from model

  Future<void> start() async {
    if (_started) return;
    _started = true;

    // Read layout once (NCHW vs NHWC) from the model
    _isNCHW = model.isNCHW; // uses your new getter

    if (!camera.value.isInitialized) {
      await camera.initialize();
    }

    if (!camera.value.isStreamingImages) {
      await camera.startImageStream((CameraImage image) {
        _latest = image; // cheap: only store the newest frame
      });
    }

    final periodMs = (1000 / maxFps).floor();
    _ticker = Timer.periodic(Duration(milliseconds: periodMs), (_) async {
      if (_busy) return;
      final imgNow = _latest;
      if (imgNow == null) return;
      _busy = true;

      try {
        // Pack planes + metadata for isolate
        final message = _packForIsolate(imgNow, _isNCHW);
        // Do all heavy work off the UI isolate (no JPEG)
        final Object inputNested = await compute<Map<String, dynamic>, Object>(
          _preprocessIsolate,
          message,
        );

        // Direct run on tensor (no decode/encode)
        final res = await model.infer(inputNested);
        if (!_controller.isClosed) _controller.add(res);
      } catch (_) {
        // keep stream alive on errors
      } finally {
        _busy = false;
      }
    });
  }

  Future<void> stop() async {
    if (!_started) return;
    _started = false;
    _ticker?.cancel();
    _ticker = null;
    _latest = null;
    if (camera.value.isStreamingImages) {
      await camera.stopImageStream();
    }
  }

  Future<void> dispose() async {
    await stop();
    await _controller.close();
  }
}

/// --------- Packing for isolate (main isolate) ----------
Map<String, dynamic> _packForIsolate(CameraImage image, bool isNCHW) {
  final w = image.width;
  final h = image.height;

  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];

  // Concatenate Y,U,V for fast transfer
  final y = yPlane.bytes, u = uPlane.bytes, v = vPlane.bytes;
  final all = Uint8List(y.length + u.length + v.length)
    ..setRange(0, y.length, y)
    ..setRange(y.length, y.length + u.length, u)
    ..setRange(y.length + u.length, y.length + u.length + v.length, v);

  final ttd = TransferableTypedData.fromList([all]);

  return {
    'w': w,
    'h': h,
    'yStride': yPlane.bytesPerRow,
    'uStride': uPlane.bytesPerRow,
    'vStride': vPlane.bytesPerRow,
    'uPixStride': uPlane.bytesPerPixel ?? 1,
    'vPixStride': vPlane.bytesPerPixel ?? 1,
    'yLen': y.length,
    'uLen': u.length,
    'isNCHW': isNCHW, // drive layout of the nested list
    'buffer': ttd,
  };
}

/// --------- Heavy preprocessing (background isolate) ----------
/// Returns `Object inputNested` matching interpreter layout:
/// - if NCHW: [[c0(HxW), c1(HxW), c2(HxW)]]  // shape [1,3,300,300]
/// - if NHWC: [ HxW x [r,g,b] ]              // shape [1,300,300,3]
Object _preprocessIsolate(Map<String, dynamic> msg) {
  final int srcW = msg['w'];
  final int srcH = msg['h'];
  final int yStride = msg['yStride'];
  final int uStride = msg['uStride'];
  final int vStride = msg['vStride'];
  final int uPixStride = msg['uPixStride'];
  final int vPixStride = msg['vPixStride'];
  final int yLen = msg['yLen'];
  final int uLen = msg['uLen'];
  final bool isNCHW = msg['isNCHW'];
  final TransferableTypedData ttd = msg['buffer'] as TransferableTypedData;

  final Uint8List all = ttd.materialize().asUint8List();
  final Uint8List y = all.sublist(0, yLen);
  final Uint8List u = all.sublist(yLen, yLen + uLen);
  final Uint8List v = all.sublist(yLen + uLen);

  // 1) YUV420 -> RGB img.Image
  final rgb = img.Image(width: srcW, height: srcH);
  int clamp(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);
  for (int row = 0; row < srcH; row++) {
    final yRow = row * yStride;
    final uRow = (row >> 1) * uStride;
    final vRow = (row >> 1) * vStride;
    for (int col = 0; col < srcW; col++) {
      final yIdx = yRow + col;
      final uIdx = uRow + (col >> 1) * uPixStride;
      final vIdx = vRow + (col >> 1) * vPixStride;
      final Y = y[yIdx];
      final U = u[uIdx];
      final V = v[vIdx];
      int r = Y + ((1436 * (V - 128)) >> 10);
      int g = Y - ((46549 * (U - 128)) >> 17) - ((93604 * (V - 128)) >> 17);
      int b = Y + ((1814 * (U - 128)) >> 10);
      rgb.setPixelRgba(col, row, clamp(r), clamp(g), clamp(b), 255);
    }
  }

  // 2) resize to 320, 3) center-crop to 300x300 (same as your predict() path)
  var im320 = img.copyResize(rgb, width: 320, height: 320);
  final off = (320 - 300) >> 1;
  final im300 = img.copyCrop(im320, x: off, y: off, width: 300, height: 300);

  // 4) normalize with PyTorch mean/std and build nested list to match layout
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  if (isNCHW) {
    // [1, 3, 300, 300]
    final c0 = List.generate(
      300,
      (_) => List<double>.filled(300, 0.0, growable: false),
      growable: false,
    );
    final c1 = List.generate(
      300,
      (_) => List<double>.filled(300, 0.0, growable: false),
      growable: false,
    );
    final c2 = List.generate(
      300,
      (_) => List<double>.filled(300, 0.0, growable: false),
      growable: false,
    );
    for (int y = 0; y < 300; y++) {
      for (int x = 0; x < 300; x++) {
        final p = im300.getPixel(x, y);
        final r = ((p.r) / 255.0 - mean[0]) / std[0];
        final g = ((p.g) / 255.0 - mean[1]) / std[1];
        final b = ((p.b) / 255.0 - mean[2]) / std[2];
        c0[y][x] = r;
        c1[y][x] = g;
        c2[y][x] = b;
      }
    }
    return [
      [c0, c1, c2],
    ];
  } else {
    // [1, 300, 300, 3]
    final hwc = List.generate(
      300,
      (_) => List.generate(
        300,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );
    for (int y = 0; y < 300; y++) {
      for (int x = 0; x < 300; x++) {
        final p = im300.getPixel(x, y);
        final r = ((p.r) / 255.0 - mean[0]) / std[0];
        final g = ((p.g) / 255.0 - mean[1]) / std[1];
        final b = ((p.b) / 255.0 - mean[2]) / std[2];
        hwc[y][x][0] = r;
        hwc[y][x][1] = g;
        hwc[y][x][2] = b;
      }
    }
    return [hwc];
  }
}
