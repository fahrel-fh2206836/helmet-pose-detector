// Class that stores the AI Model (tflite) and methods for inferencin

import 'dart:math' as math;
import 'package:tflite_flutter/tflite_flutter.dart';

class HelmetPose {
  // Classes that the model classifies (index 0 -> looking && index 1 -> not_looking)
  static const classes = ["looking", "not_looking"];

  // Under-the-hood TFLite interpreter object (loads and runs the model).
  final Interpreter _interpreter;

  HelmetPose(this._interpreter);

  // Initialize and loads interpreter with AI model (tflite from assets) with NNAPI and threads configurations
  static Future<HelmetPose> load({
    String assetPath = 'assets/ghostnet_100_fp32io_fp16.tflite',
    int threads = 4,
  }) async {
    final options = InterpreterOptions()
      ..addDelegate(
        XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: threads)),
      );
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

  int get modelHeight {
    final inShape = _interpreter.getInputTensor(0).shape;
    if (isNCHW) {
      return inShape[2];
    } else {
      return inShape[1];
    }
  }

  int get modelWidth {
    final inShape = _interpreter.getInputTensor(0).shape;
    if (isNCHW) {
      return inShape[3];
    } else {
      return inShape[2];
    }
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
