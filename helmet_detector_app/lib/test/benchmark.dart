import 'dart:math';
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';

/// ----------------------------------------------
/// Delegate benchmark utility for TensorFlow Lite
/// ----------------------------------------------
///
/// This script compares:
/// 1️⃣ Default (no delegate) — runs with XNNPACK by default
/// 2️⃣ Plain CPU (no XNNPACK)
/// 3️⃣ NNAPI delegate
///
/// Prints average inference time, min/max, and FPS.
/// Works for FP16 or INT8 models.
/// ----------------------------------------------

class DelegateBenchmark {
  final String modelPath;
  final int threads;

  DelegateBenchmark({required this.modelPath, this.threads = 4});

  /// Minimal preprocessing stub:
  /// replace with your real preprocess function if needed.
  Object preprocess(Uint8List bytes, Interpreter itp) {
    // Just use a zero tensor shaped like the model’s input
    final inputShape = itp.getInputTensor(0).shape;
    final inputType = itp.getInputTensor(0).type;

    if (inputType == TensorType.float32) {
      return List.generate(
        inputShape[0],
        (_) => List.generate(
          inputShape[1],
          (_) => List.filled(inputShape.last, 0.0),
        ),
      );
    } else if (inputType == TensorType.int8) {
      return List.generate(
        inputShape[0],
        (_) => List.generate(
          inputShape[1],
          (_) => List.filled(inputShape.last, 0),
        ),
      );
    } else {
      throw StateError('Unsupported tensor type: $inputType');
    }
  }

  Future<void> benchmarkAll() async {
    print('🚀 Starting delegate benchmarks on model: $modelPath\n');

    // Load a dummy image or tensor input (optional)
    final dummyInput = Uint8List(1);

    await _runBenchmark(
      name: 'Plain',
      options: InterpreterOptions()..threads = threads,
      dummyInput: dummyInput,
    );

    await _runBenchmark(
      name: 'XNNPACK',
      options: (InterpreterOptions()
        ..threads = threads
        ..addDelegate(XNNPackDelegate())),
      dummyInput: dummyInput,
    );

    await _runBenchmark(
      name: 'NNAPI',
      options: (InterpreterOptions()
        ..threads = threads
        ..useNnApiForAndroid = true),
      dummyInput: dummyInput,
    );

    print('✅ Benchmark complete.');
  }

  Future<void> _runBenchmark({
    required String name,
    required InterpreterOptions options,
    required Uint8List dummyInput,
    int runs = 30,
  }) async {
    print('🔹 Testing delegate: $name');

    // Load interpreter
    final itp = await Interpreter.fromAsset(modelPath, options: options);

    // Prepare input/output buffers
    final input = preprocess(dummyInput, itp);
    final outputShape = itp.getOutputTensor(0).shape;
    final output = List.generate(
      outputShape[0],
      (_) => List.filled(outputShape[1], 0),
    );

    // Warm-up
    itp.run(input, output);

    final times = <double>[];
    for (int i = 0; i < runs; i++) {
      final sw = Stopwatch()..start();
      itp.run(input, output);
      sw.stop();
      times.add(sw.elapsedMicroseconds / 1000.0);
    }

    final avg = times.reduce((a, b) => a + b) / times.length;
    final minT = times.reduce(min);
    final maxT = times.reduce(max);
    final fps = 1000 / avg;

    print(
      '✅ $name | avg=${avg.toStringAsFixed(2)} ms '
      '(min=${minT.toStringAsFixed(1)}, max=${maxT.toStringAsFixed(1)}) | '
      'FPS≈${fps.toStringAsFixed(1)}\n',
    );

    itp.close();
  }
}

/// Example entrypoint for testing
Future<void> runDelegateBenchmarks() async {
  final bench = DelegateBenchmark(modelPath: 'assets/helmet_pose_fp16.tflite');
  await bench.benchmarkAll();
}
