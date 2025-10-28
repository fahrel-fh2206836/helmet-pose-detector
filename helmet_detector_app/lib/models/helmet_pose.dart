// Class that stores the AI Model (tflite) and methods for inferencin

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class HelmetPose {
  // Classes that the model classifies (index 0 -> looking && index 1 -> not_looking)
  static const classes = ["looking", "not_looking"];

  // Height, width, channels (RGB) that matches model input requirements
  static const int H = 300, W = 300, C = 3;

  // Under-the-hood TFLite interpreter object (loads and runs the model).
  final Interpreter _interpreter;

  HelmetPose(this._interpreter);

  // Initialize and loads interpreter with AI model (tflite from assets) with NNAPI and threads configurations
  static Future<HelmetPose> load({
    String assetPath = 'assets/helmet_pose_fp32io_fp16.tflite',
    int threads = 4,
  }) async {
    final options = InterpreterOptions()..threads = threads;
    options.useNnApiForAndroid = true;
    final itp = await Interpreter.fromAsset(assetPath, options: options);
    return HelmetPose(itp);
  }

  // Stops running the interpreter to release resources
  void close() => _interpreter.close();

  /* 
  Expose shapes once so the app can know if input is NCHW vs NHWC.
  N -> Number of samples
  C -> Channels
  H -> Height
  W -> Width
  e.g., [1,3,300,300] (NCHW) vs [1,300,300,3] (NHWC)
  */
  bool get isNCHW {
    final inShape = _interpreter.getInputTensor(0).shape;
    return inShape.length == 4 && inShape[0] == 1 && inShape[1] == 3;
  }

  /* 
  Predict from raw image bytes (camera/gallery).
  Returns (label, prob) with prob in [0..1] (softmax confidence).
  Use when preprocessing isn't performed on streamed images and input for model is not formatted yet.
  (Used in benchmarking/testings)
   */
  Future<({String label, double prob})> classifyImage(Uint8List bytes) async {
    // Decode bytes → resize → center-crop to 300x300 (preprocessing)
    img.Image? im = img.decodeImage(bytes);
    if (im == null) throw StateError("Unsupported image data.");
    im = img.copyResize(im, width: 320, height: 320);
    final off = (320 - 300) ~/ 2;
    im = img.copyCrop(im, x: off, y: off, width: 300, height: 300);

    //PyTorch normalization constants for EfficientNetB3 model
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // This will hold the nested Dart Lists in the exact shape the interpreter expects.
    late final Object inputNested;

    // Allocate 3 separate 2D arrays, one per channel (R, G, B).
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

      // Fill each channel with normalized float values.
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          // read pixel at (x,y)
          final img.Color c = im.getPixel(x, y);

          // scale to [0,1]
          final r = c.r.toDouble() / 255.0;
          final g = c.g.toDouble() / 255.0;
          final b = c.b.toDouble() / 255.0;

          // normalize
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

      // Allocate H×W×3 (per pixel: [R,G,B]).
      final hwc = List.generate(
        H,
        (_) => List.generate(
          W,
          (_) => List<double>.filled(3, 0.0, growable: false),
          growable: false,
        ),
        growable: false,
      );

      // Fill HWC with normalized float values.
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          // read pixel at (x,y)
          final img.Color c = im.getPixel(x, y);

          // scale to [0,1]
          final r = c.r.toDouble() / 255.0;
          final g = c.g.toDouble() / 255.0;
          final b = c.b.toDouble() / 255.0;

          // normalize
          hwc[y][x][0] = (r - mean[0]) / std[0];
          hwc[y][x][1] = (g - mean[1]) / std[1];
          hwc[y][x][2] = (b - mean[2]) / std[2];
        }
      }
      inputNested = [hwc];
    }

    // Prepare output buffer [1,2] and run
    final outShape = _interpreter.getOutputTensor(0).shape; // e.g. [1,2]
    final output = List.generate(
      outShape[0],
      (_) => List<double>.filled(outShape[1], 0.0, growable: false),
      growable: false,
    );

    // Run inference: fills `output` in place.
    _interpreter.run(inputNested, output);

    // Read raw scores (logits) for 2 classes and compute stable softmax.
    final a = output[0][0];
    final b = output[0][1];
    final m = a > b ? a : b;
    final ea = math.exp(a - m);
    final eb = math.exp(b - m);
    final s = ea + eb;

    // prob for class 0 and 1, respectively.
    final p0 = ea / s, p1 = eb / s;

    // Choose the larger prob and return its label + probability.
    final predIdx = p1 > p0 ? 1 : 0;
    final prob = predIdx == 0 ? p0 : p1;
    return (label: classes[predIdx], prob: prob);
  }

  /*
  Note: run when you already have a nested input tensor prepared.
  `inputNested` must match the interpreter's expected layout.
  Returns (label, prob) and applies the same softmax.
  Use: When tensor input is formatted so method just runs inference 
  (Used within app's logic to avoid heavy preprocessing on main thread)
  */
  Future<({String label, double prob})> runInference(Object inputNested) async {
    // Read output shape, e.g., [1,2]
    final outShape = _interpreter.getOutputTensor(0).shape;

    // Allocate output based on the model's output shape.
    final output = List.generate(
      outShape[0],
      (_) => List<double>.filled(outShape[1], 0.0, growable: false),
      growable: false,
    );

    // Run the model with your prepared input and stores output in 'output' variable.
    _interpreter.run(inputNested, output);

    // Read raw scores (logits) for 2 classes and compute stable softmax.
    final a = output[0][0], b = output[0][1];
    final m = math.max(a, b);
    final ea = math.exp(a - m), eb = math.exp(b - m);
    final s = ea + eb;

    // prob for class 0 and 1, respectively.
    final p0 = ea / s, p1 = eb / s;

    // Pick the top class and output label + confidence.
    final predIdx = p1 > p0 ? 1 : 0;
    final prob = predIdx == 0 ? p0 : p1;
    return (label: HelmetPose.classes[predIdx], prob: prob);
  }
}
