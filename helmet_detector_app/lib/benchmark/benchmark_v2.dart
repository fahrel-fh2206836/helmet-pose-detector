// benchmark_isolate.dart
import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';

/// -------------------- Public entrypoint (call this from UI) --------------------
Future<void> runBenchmarkSuiteIsolated({
  required String deviceName,
  required String modelALabel, // e.g. "FP32"
  required String modelAAsset, // e.g. "assets/helmet_pose_fp32.tflite"
  required String modelBLabel, // e.g. "FP32 I/O + FP16 internal"
  required String
  modelBAsset, // e.g. "assets/helmet_pose_fp32io_fp16internal.tflite"
  String sampleImageAsset = 'assets/sample.png',
  List<int> threads = const [1, 2, 3, 4, 6, 8],
  int runs = 30,
  int warmup = 5,
}) async {
  // Prepare assets on MAIN isolate
  final modelAPath = await _copyAssetToTemp(modelAAsset);
  final modelBPath = await _copyAssetToTemp(modelBAsset);
  final imageBytes = (await rootBundle.load(
    sampleImageAsset,
  )).buffer.asUint8List();

  // Run in background isolate
  final markdown = await Isolate.run(() async {
    return await _workerBenchmarkSuite(
      deviceName: deviceName,
      modelALabel: modelALabel,
      modelAPath: modelAPath,
      modelBLabel: modelBLabel,
      modelBPath: modelBPath,
      imageBytes: imageBytes,
      threads: threads,
      runs: runs,
      warmup: warmup,
    );
  });

  print(markdown);
}

/// Copy a bundled asset to a temp file so we can open it in the worker isolate.
Future<String> _copyAssetToTemp(String assetPath) async {
  final dir = await getTemporaryDirectory();
  final outFile = File('${dir.path}/${assetPath.split('/').last}');
  final data = await rootBundle.load(assetPath);
  await outFile.writeAsBytes(
    data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes),
    flush: true,
  );
  return outFile.path;
}

/// -------------------- Worker isolate code --------------------

class _BenchStats {
  final double avg, minT, median, p90, maxT, fps;
  _BenchStats({
    required this.avg,
    required this.minT,
    required this.median,
    required this.p90,
    required this.maxT,
  }) : fps = 1000.0 / avg;

  String toCell() =>
      "${avg.toStringAsFixed(2)} / ${minT.toStringAsFixed(1)} / "
      "${median.toStringAsFixed(1)} / ${p90.toStringAsFixed(1)} / "
      "${maxT.toStringAsFixed(1)} / ≈${fps.toStringAsFixed(1)} FPS";
}

InterpreterOptions _optsXnn(int threads) => (InterpreterOptions()
  ..threads = threads
  ..addDelegate(XNNPackDelegate())
  ..useNnApiForAndroid = false);
InterpreterOptions _optsCpu(int threads) => (InterpreterOptions()
  ..threads = threads
  ..useNnApiForAndroid = false);
InterpreterOptions _optsNnapi() =>
    (InterpreterOptions()..useNnApiForAndroid = true);

double _percentile(List<double> sorted, double q) {
  final n = sorted.length;
  final pos = (q * (n - 1));
  final i = pos.floor();
  final frac = pos - i;
  if (i >= n - 1) return sorted.last;
  return sorted[i] + frac * (sorted[i + 1] - sorted[i]);
}

Object _buildInput({
  required Uint8List bytes,
  required bool isNCHW,
  required int H,
  required int W,
}) {
  img.Image? im = img.decodeImage(bytes);
  if (im == null) {
    throw StateError("Unsupported image data.");
  }
  im = img.copyResize(im, width: 320, height: 320);
  final off = (320 - W) ~/ 2;
  im = img.copyCrop(im, x: off, y: off, width: W, height: H);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  if (isNCHW) {
    final c0 = List.generate(
      H,
      (_) => List<double>.filled(W, 0.0),
      growable: false,
    );
    final c1 = List.generate(
      H,
      (_) => List<double>.filled(W, 0.0),
      growable: false,
    );
    final c2 = List.generate(
      H,
      (_) => List<double>.filled(W, 0.0),
      growable: false,
    );
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        final c = im.getPixel(x, y);
        final r = c.r.toDouble() / 255.0;
        final g = c.g.toDouble() / 255.0;
        final b = c.b.toDouble() / 255.0;
        c0[y][x] = (r - mean[0]) / std[0];
        c1[y][x] = (g - mean[1]) / std[1];
        c2[y][x] = (b - mean[2]) / std[2];
      }
    }
    return [
      [c0, c1, c2],
    ];
  } else {
    final hwc = List.generate(
      H,
      (_) =>
          List.generate(W, (_) => List<double>.filled(3, 0.0), growable: false),
      growable: false,
    );
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        final c = im.getPixel(x, y);
        final r = c.r.toDouble() / 255.0;
        final g = c.g.toDouble() / 255.0;
        final b = c.b.toDouble() / 255.0;
        hwc[y][x][0] = (r - mean[0]) / std[0];
        hwc[y][x][1] = (g - mean[1]) / std[1];
        hwc[y][x][2] = (b - mean[2]) / std[2];
      }
    }
    return [hwc];
  }
}

Future<_BenchStats> _runOne({
  required String modelPath,
  required Uint8List imageBytes,
  required InterpreterOptions options,
  int runs = 30,
  int warmup = 5,
}) async {
  final itp = Interpreter.fromFile(File(modelPath), options: options);
  itp.allocateTensors();

  final inShape = itp.getInputTensor(0).shape;
  late final bool isNCHW;
  late final int H, W;
  if (inShape.length == 4 && inShape[1] == 3) {
    isNCHW = true;
    H = inShape[2];
    W = inShape[3];
  } else if (inShape.length == 4 && inShape[3] == 3) {
    isNCHW = false;
    H = inShape[1];
    W = inShape[2];
  } else {
    itp.close();
    throw StateError('Unsupported input shape: $inShape');
  }

  final input = _buildInput(bytes: imageBytes, isNCHW: isNCHW, H: H, W: W);

  // Warmup
  for (int i = 0; i < warmup; i++) {
    await runInference(input, itp);
  }

  final times = <double>[];
  for (int i = 0; i < runs; i++) {
    final sw = Stopwatch()..start();
    await runInference(input, itp);
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }

  times.sort();
  final avg = times.reduce((a, b) => a + b) / times.length;
  final minT = times.first;
  final maxT = times.last;
  final median = times.length.isOdd
      ? times[times.length ~/ 2]
      : (times[times.length ~/ 2 - 1] + times[times.length ~/ 2]) / 2.0;
  final p90 = _percentile(times, 0.90);

  itp.close();

  return _BenchStats(
    avg: avg,
    minT: minT,
    median: median,
    p90: p90,
    maxT: maxT,
  );
}

Future<({String label, double prob})> runInference(
  Object inputNested,
  Interpreter interpreter,
) async {
  // Read output shape, e.g., [1,2]
  final outShape = interpreter.getOutputTensor(0).shape;

  // Allocate output based on the model's output shape.
  final output = List.generate(
    outShape[0],
    (_) => List<double>.filled(outShape[1], 0.0, growable: false),
    growable: false,
  );

  // Run the model with your prepared input and stores output in 'output' variable.
  interpreter.run(inputNested, output);

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

/// Safe wrapper to prevent a single failure from stopping the suite
Future<_BenchStats?> _safeRunOne({
  required String modelPath,
  required Uint8List imageBytes,
  required InterpreterOptions options,
  int runs = 30,
  int warmup = 5,
  String debugTag = '',
}) async {
  try {
    return await _runOne(
      modelPath: modelPath,
      imageBytes: imageBytes,
      options: options,
      runs: runs,
      warmup: warmup,
    );
  } catch (e, st) {
    print('[bench] $debugTag failed: $e\n$st');
    return null;
  }
}

/// -------------------- Main benchmark suite --------------------
Future<String> _workerBenchmarkSuite({
  required String deviceName,
  required String modelALabel,
  required String modelAPath,
  required String modelBLabel,
  required String modelBPath,
  required Uint8List imageBytes,
  required List<int> threads,
  required int runs,
  required int warmup,
}) async {
  final Map<String, Map<int?, Map<String, _BenchStats>>> store = {
    'XNNPACK': {},
    'CPU': {},
    'NNAPI': {},
  };

  final int cpuCount = Platform.numberOfProcessors;
  final List<int> clampedThreads =
      (threads.map((t) => (t.clamp(1, cpuCount) as int)).toSet().toList()
        ..sort());

  // // XNNPACK sweep
  // for (final t in clampedThreads) {
  //   final a = await _safeRunOne(
  //     modelPath: modelAPath,
  //     imageBytes: imageBytes,
  //     options: _optsXnn(t),
  //     runs: runs,
  //     warmup: warmup,
  //     debugTag: 'XNNPACK/$t $modelALabel',
  //   );
  //   final b = await _safeRunOne(
  //     modelPath: modelBPath,
  //     imageBytes: imageBytes,
  //     options: _optsXnn(t),
  //     runs: runs,
  //     warmup: warmup,
  //     debugTag: 'XNNPACK/$t $modelBLabel',
  //   );
  //   if (a != null || b != null) {
  //     store['XNNPACK']![t] = {
  //       if (a != null) modelALabel: a,
  //       if (b != null) modelBLabel: b,
  //     };
  //   }
  // }

  // // CPU sweep
  // for (final t in clampedThreads) {
  //   final a = await _safeRunOne(
  //     modelPath: modelAPath,
  //     imageBytes: imageBytes,
  //     options: _optsCpu(t),
  //     runs: runs,
  //     warmup: warmup,
  //     debugTag: 'CPU/$t $modelALabel',
  //   );
  //   final b = await _safeRunOne(
  //     modelPath: modelBPath,
  //     imageBytes: imageBytes,
  //     options: _optsCpu(t),
  //     runs: runs,
  //     warmup: warmup,
  //     debugTag: 'CPU/$t $modelBLabel',
  //   );
  //   if (a != null || b != null) {
  //     store['CPU']![t] = {
  //       if (a != null) modelALabel: a,
  //       if (b != null) modelBLabel: b,
  //     };
  //   }
  // }

  // NNAPI (best-effort)
  _BenchStats? aN;
  _BenchStats? bN;
  try {
    aN = await _safeRunOne(
      modelPath: modelAPath,
      imageBytes: imageBytes,
      options: _optsNnapi(),
      runs: runs,
      warmup: warmup,
      debugTag: 'NNAPI $modelALabel',
    );
    bN = await _safeRunOne(
      modelPath: modelBPath,
      imageBytes: imageBytes,
      options: _optsNnapi(),
      runs: runs,
      warmup: warmup,
      debugTag: 'NNAPI $modelBLabel',
    );
  } catch (_) {}
  if (aN != null || bN != null) {
    store['NNAPI']![null] = {
      if (aN != null) modelALabel: aN,
      if (bN != null) modelBLabel: bN,
    };
  }

  String cell(_BenchStats? s) => s == null ? "N/A" : s.toCell();

  final buf = StringBuffer()
    ..writeln("### 📊 TFLite Inference Benchmark — $deviceName")
    ..writeln()
    ..writeln(
      "| **Delegate** | **Threads** | **$modelALabel**<br>Avg / Min / Median / P90 / Max / FPS | **$modelBLabel**<br>Avg / Min / Median / P90 / Max / FPS |",
    )
    ..writeln(
      "|:-------------|:-----------:|:----------------------------------------------------------|:-----------------------------------------------------------|",
    );

  void rows(String delegate) {
    final entries = store[delegate]!;
    final keys =
        entries.keys
            .where((k) => k != null && entries[k]!.isNotEmpty)
            .cast<int>()
            .toList()
          ..sort();
    for (final t in keys) {
      final a = entries[t]?[modelALabel];
      final b = entries[t]?[modelBLabel];
      buf.writeln("| **$delegate** | $t | ${cell(a)} | ${cell(b)} |");
    }
  }

  // rows('XNNPACK');
  // rows('CPU');

  if (store['NNAPI']![null] != null && store['NNAPI']![null]!.isNotEmpty) {
    final aL = store['NNAPI']![null]?[modelALabel];
    final bL = store['NNAPI']![null]?[modelBLabel];
    buf.writeln("| **NNAPI** | – | ${cell(aL)} | ${cell(bL)} |");
  }

  final clampedThreadList = clampedThreads.join(', ');
  buf
    ..writeln()
    ..writeln(
      "**Notes:** Runs=$runs, Warmup=$warmup. "
      "XNNPACK/CPU use threads=$clampedThreadList (clamped to $cpuCount cores). "
      "NNAPI ignores threads and is shown last.",
    );

  return buf.toString();
}
