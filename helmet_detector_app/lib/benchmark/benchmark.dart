// File used for checking latency on deferent delegation on which the AI model runs on.
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// Prepares the interpreter settings for benchmarking
Future<void> benchmarkHelmetPose({
  required String name,
  required String assetPath,
  required Uint8List sampleImageBytes,
  InterpreterOptions? options,
  int runs = 20,
}) async {
  print('🔹 Benchmarking mode: $name');

  final itp = await Interpreter.fromAsset(assetPath, options: options);
  final newPose = HelmetPose(itp);
  _runBench(name, newPose, sampleImageBytes, runs);
}

// Runs and Calculates the benchmarking metrics
Future<void> _runBench(
  String name,
  HelmetPose pose,
  Uint8List bytes,
  int runs,
) async {
  await pose.classifyImage(bytes);

  // Stores list of time period it takes to complete an inference.
  final times = <double>[];

  // Repeated iteration of helmet pose prediction
  for (int i = 0; i < runs; i++) {
    final sw = Stopwatch()..start();
    await pose.classifyImage(bytes);
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }

  // Metrics results
  final avg = times.reduce((a, b) => a + b) / times.length;
  final minT = times.reduce(math.min);
  final maxT = times.reduce(math.max);
  final fps = 1000 / avg;

  print(
    '✅ $name | avg=${avg.toStringAsFixed(2)} ms '
    '(min=${minT.toStringAsFixed(1)}, max=${maxT.toStringAsFixed(1)}) '
    '| FPS≈${fps.toStringAsFixed(1)}\n',
  );

  pose.close();
}

// Call this method to start benchmarking
Future<void> testDelegates() async {
  final bytes = await rootBundle.load('assets/sample.png');
  final imageBytes = bytes.buffer.asUint8List();
  // const modelPath = 'assets/helmet_pose_fp32io_fp16.tflite';
  const modelPath = 'assets/helmet_pose_fp32.tflite';

  final xnnpackOpts = InterpreterOptions()
    ..threads = 4
    ..addDelegate(XNNPackDelegate());
  await benchmarkHelmetPose(
    name: 'XNNPACK',
    assetPath: modelPath,
    sampleImageBytes: imageBytes,
    options: xnnpackOpts,
  );

  final nnapiOpts = InterpreterOptions()
    ..threads = 4
    ..useNnApiForAndroid = true;
  await benchmarkHelmetPose(
    name: 'NNAPI delegate',
    assetPath: modelPath,
    sampleImageBytes: imageBytes,
    options: nnapiOpts,
  );

  final gpuOpts = InterpreterOptions()
    ..threads = 4
    ..addDelegate(
      GpuDelegateV2(
        options: GpuDelegateOptionsV2(
          isPrecisionLossAllowed: true, // enables FP16 path
        ),
      ),
    );
  await benchmarkHelmetPose(
    name: 'GPUV2 delegate',
    assetPath: modelPath,
    sampleImageBytes: imageBytes,
    options: gpuOpts,
  );

  final defaultOpts = InterpreterOptions()..threads = 4;
  await benchmarkHelmetPose(
    name: 'Default',
    assetPath: modelPath,
    sampleImageBytes: imageBytes,
    options: defaultOpts,
  );
}
