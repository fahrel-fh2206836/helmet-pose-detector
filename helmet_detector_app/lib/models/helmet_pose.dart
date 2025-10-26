import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class HelmetPose {
  static const classes = ["looking", "not_looking"];
  static const int H = 300, W = 300, C = 3;

  final Interpreter _interpreter;

  HelmetPose(this._interpreter);

  static Future<HelmetPose> load({
    String assetPath = 'assets/helmet_pose_fp16.tflite',
    int threads = 4,
  }) async {
    final options = InterpreterOptions()..threads = threads;
    options.useNnApiForAndroid = true;
    final itp = await Interpreter.fromAsset(assetPath, options: options);
    return HelmetPose(itp);
  }

  void close() => _interpreter.close();

  /// Predict from raw image bytes (camera/gallery).
  /// Returns (label, prob) with prob in [0..1].
  Future<({String label, double prob})> predict(Uint8List bytes) async {
    // --- Decode → resize → center-crop to 300x300 ---
    img.Image? im = img.decodeImage(bytes);
    if (im == null) throw StateError("Unsupported image data.");
    im = img.copyResize(im, width: 320, height: 320);
    final off = (320 - 300) ~/ 2;
    im = img.copyCrop(im, x: off, y: off, width: 300, height: 300);

    // --- Read input/output shapes ---
    final inShape = _interpreter
        .getInputTensor(0)
        .shape; // e.g. [1,3,300,300] or [1,300,300,3]
    final outShape = _interpreter.getOutputTensor(0).shape; // e.g. [1,2]
    final isNCHW = inShape.length == 4 && inShape[0] == 1 && inShape[1] == 3;

    // --- PyTorch normalization constants ---
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // --- Build nested input List for tflite_flutter 0.11.0 `run()` ---
    late final Object inputNested;

    if (isNCHW) {
      // Make [1, 3, 300, 300]
      final c0 = List.generate(
        H,
        (_) => List<double>.filled(W, 0.0, growable: false),
        growable: false,
      );
      final c1 = List.generate(
        H,
        (_) => List<double>.filled(W, 0.0, growable: false),
        growable: false,
      );
      final c2 = List.generate(
        H,
        (_) => List<double>.filled(W, 0.0, growable: false),
        growable: false,
      );

      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          final img.Color c = im.getPixel(x, y);
          final r = c.r.toDouble() / 255.0;
          final g = c.g.toDouble() / 255.0;
          final b = c.b.toDouble() / 255.0;
          c0[y][x] = (r - mean[0]) / std[0];
          c1[y][x] = (g - mean[1]) / std[1];
          c2[y][x] = (b - mean[2]) / std[2];
        }
      }
      inputNested = [
        [c0, c1, c2],
      ];
    } else {
      // Assume NHWC: [1, 300, 300, 3]
      final hwc = List.generate(
        H,
        (_) => List.generate(
          W,
          (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      );
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          final img.Color c = im.getPixel(x, y);
          final r = c.r.toDouble() / 255.0;
          final g = c.g.toDouble() / 255.0;
          final b = c.b.toDouble() / 255.0;
          hwc[y][x][0] = (r - mean[0]) / std[0];
          hwc[y][x][1] = (g - mean[1]) / std[1];
          hwc[y][x][2] = (b - mean[2]) / std[2];
        }
      }
      inputNested = [hwc];
    }

    // --- Prepare output buffer [1,2] and run ---
    final output = List.generate(
      outShape[0],
      (_) => List<double>.filled(outShape[1], 0.0, growable: false),
      growable: false,
    );

    _interpreter.run(inputNested, output);

    // --- Read logits and softmax (2 classes) ---
    final a = output[0][0];
    final b = output[0][1];
    final m = a > b ? a : b;
    final ea = math.exp(a - m);
    final eb = math.exp(b - m);
    final s = ea + eb;
    final p0 = ea / s, p1 = eb / s;

    final predIdx = p1 > p0 ? 1 : 0;
    final prob = predIdx == 0 ? p0 : p1;
    return (label: classes[predIdx], prob: prob);
  }

  /// Expose shapes once so the app can know if input is NCHW vs NHWC.
  bool get isNCHW {
    final inShape = _interpreter.getInputTensor(0).shape;
    // e.g., [1,3,300,300] vs [1,300,300,3]
    return inShape.length == 4 && inShape[0] == 1 && inShape[1] == 3;
  }

  /// Convenience: run when you already have a nested input tensor prepared.
  /// `inputNested` must match the interpreter's expected layout.
  /// Returns (label, prob) and applies the same softmax you already use.
  Future<({String label, double prob})> infer(Object inputNested) async {
    // Read output shape, e.g., [1,2]
    final outShape = _interpreter.getOutputTensor(0).shape;
    final output = List.generate(
      outShape[0],
      (_) => List<double>.filled(outShape[1], 0.0, growable: false),
      growable: false,
    );

    _interpreter.run(inputNested, output); // same as in predict()
    final a = output[0][0], b = output[0][1];
    final m = math.max(a, b);
    final ea = math.exp(a - m), eb = math.exp(b - m);
    final s = ea + eb;
    final p0 = ea / s, p1 = eb / s;
    final predIdx = p1 > p0 ? 1 : 0;
    final prob = predIdx == 0 ? p0 : p1;
    return (label: HelmetPose.classes[predIdx], prob: prob);
  }
}
