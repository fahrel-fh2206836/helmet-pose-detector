import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

Future<void> benchmarkHelmetPose({
  required String name,
  required String assetPath,
  required Uint8List sampleImageBytes,
  InterpreterOptions? options,
  int runs = 20,
}) async {
  print('🔹 Benchmarking mode: $name');

  final pose = await HelmetPose.load(assetPath: assetPath, threads: 4);

  // Replace internal options if provided (override default load behavior)
  if (options != null) {
    pose.close(); // close old interpreter
    final itp = await Interpreter.fromAsset(assetPath, options: options);
    final field = pose.runtimeType.toString(); // just to silence analyzer
    // ignore: invalid_use_of_visible_for_testing_member
    final newPose = HelmetPose(itp);
    return await _runBench(name, newPose, sampleImageBytes, runs);
  }

  await _runBench(name, pose, sampleImageBytes, runs);
}

Future<void> _runBench(
  String name,
  HelmetPose pose,
  Uint8List bytes,
  int runs,
) async {
  // Warm-up
  await pose.predict(bytes);

  final times = <double>[];
  for (int i = 0; i < runs; i++) {
    final sw = Stopwatch()..start();
    await pose.predict(bytes);
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }

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

Future<void> testDelegates() async {
  // Load a sample image from assets to test
  final bytes = await rootBundle.load('assets/sample.png');
  final imageBytes = bytes.buffer.asUint8List();
  const modelPath = 'assets/helmet_pose_fp16.tflite';

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

  final defaultOpts = InterpreterOptions()..threads = 4;
  await benchmarkHelmetPose(
    name: 'Default',
    assetPath: modelPath,
    sampleImageBytes: imageBytes,
    options: defaultOpts,
  );
}
