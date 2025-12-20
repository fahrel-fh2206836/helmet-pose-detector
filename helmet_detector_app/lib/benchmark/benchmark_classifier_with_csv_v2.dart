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
Future<void> runBenchmarkSuiteIsolatedWithCSV({
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
  String selectDelegate = 'XNNPACK',
}) async {
  // Prepare assets on MAIN isolate
  final modelAPath = await _copyAssetToTemp(modelAAsset);
  final modelBPath = await _copyAssetToTemp(modelBAsset);
  final imageBytes = (await rootBundle.load(
    sampleImageAsset,
  )).buffer.asUint8List();

  // Decide where to save CSV (e.g. app documents dir)
  final docsDir = await getApplicationDocumentsDirectory();
  final csvPath = '${docsDir.path}/benchmark_${deviceName}_$selectDelegate.csv';

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
      selectDelegate: selectDelegate,
      csvOutputPath: csvPath,
    );
  });

  print(markdown);
  print("Benchmark Completed for $selectDelegate");
  print("CSV saved at: $csvPath");
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
  ..addDelegate(
    XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: threads)),
  )
  ..useNnApiForAndroid = false);
InterpreterOptions _optsCpu(int threads) => (InterpreterOptions()
  ..threads = threads
  ..useNnApiForAndroid = false);
InterpreterOptions _optsNnapi() =>
    (InterpreterOptions()..useNnApiForAndroid = true);
InterpreterOptions _optsGpu() {
  final gpu = GpuDelegateV2(
    options: GpuDelegateOptionsV2(
      isPrecisionLossAllowed: true, // enables FP16 on GPU
    ),
  );
  return (InterpreterOptions()
    ..addDelegate(gpu)
    ..useNnApiForAndroid = false);
}

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
  required Interpreter interpreter,
}) {
  img.Image? im = img.decodeImage(bytes);
  if (im == null) throw StateError("Unsupported image data.");

  im = img.copyResize(im, width: W, height: H);

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  // --- First build FP32-normalized tensor shapes ---
  Object fpInput;
  if (isNCHW) {
    final c0 = List.generate(H, (_) => List<double>.filled(W, 0.0));
    final c1 = List.generate(H, (_) => List<double>.filled(W, 0.0));
    final c2 = List.generate(H, (_) => List<double>.filled(W, 0.0));

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        final p = im.getPixel(x, y);
        final r = p.r / 255.0;
        final g = p.g / 255.0;
        final b = p.b / 255.0;

        c0[y][x] = (r - mean[0]) / std[0];
        c1[y][x] = (g - mean[1]) / std[1];
        c2[y][x] = (b - mean[2]) / std[2];
      }
    }
    fpInput = [
      [c0, c1, c2],
    ];
  } else {
    final hwc = List.generate(
      H,
      (_) => List.generate(W, (_) => List<double>.filled(3, 0.0)),
    );

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        final p = im.getPixel(x, y);
        final r = p.r / 255.0;
        final g = p.g / 255.0;
        final b = p.b / 255.0;

        hwc[y][x][0] = (r - mean[0]) / std[0];
        hwc[y][x][1] = (g - mean[1]) / std[1];
        hwc[y][x][2] = (b - mean[2]) / std[2];
      }
    }
    fpInput = [hwc];
  }

  // --- Now check input tensor dtype ---
  final inputTensor = interpreter.getInputTensor(0);
  final type = inputTensor.type;

  if (type == TensorType.float32 || type == TensorType.float16) {
    // No quantization needed
    return fpInput;
  }

  if (type == TensorType.int8) {
    final scale = inputTensor.params.scale;
    final zero = inputTensor.params.zeroPoint;

    // Quantize FP32 → INT8
    Object quantize(Object x) {
      if (x is double) {
        final q = (x / scale + zero).round().clamp(-128, 127);
        return q; // int
      } else if (x is List) {
        return x.map((v) => quantize(v)).toList();
      } else {
        throw StateError("Unsupported type in quantization");
      }
    }

    return quantize(fpInput);
  }

  throw StateError("Unsupported input type: $type");
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

  final input = _buildInput(
    bytes: imageBytes,
    isNCHW: isNCHW,
    H: H,
    W: W,
    interpreter: itp,
  );

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
  final outTensor = interpreter.getOutputTensor(0);
  final outType = outTensor.type;

  final outShape = outTensor.shape;

  // Allocate FP32 structure to store final logits
  final output = List.generate(
    outShape[0],
    (_) => List<double>.filled(outShape[1], 0.0),
  );

  // Temporary storage for raw output
  Object rawOut;

  if (outType == TensorType.int8) {
    // Prepare int8 container
    rawOut = List.generate(
      outShape[0],
      (_) => List<int>.filled(outShape[1], 0),
    );
  } else {
    rawOut = List.generate(
      outShape[0],
      (_) => List<double>.filled(outShape[1], 0.0),
    );
  }

  // Run inference
  interpreter.run(inputNested, rawOut);

  // --- Dequantize if needed ---
  if (outType == TensorType.int8) {
    final scale = outTensor.params.scale;
    final zero = outTensor.params.zeroPoint;

    for (int i = 0; i < outShape[0]; i++) {
      for (int j = 0; j < outShape[1]; j++) {
        output[i][j] = ((rawOut as List<List<int>>)[i][j] - zero) * scale;
      }
    }
  } else {
    // Already FP32
    for (int i = 0; i < outShape[0]; i++) {
      for (int j = 0; j < outShape[1]; j++) {
        output[i][j] = (rawOut as List<List<double>>)[i][j];
      }
    }
  }

  // Apply softmax (stable)
  final a = output[0][0], b = output[0][1];
  final m = math.max(a, b);
  final ea = math.exp(a - m), eb = math.exp(b - m);
  final s = ea + eb;
  final p0 = ea / s, p1 = eb / s;

  final predIdx = p1 > p0 ? 1 : 0;
  final prob = predIdx == 1 ? p1 : p0;

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
  required String selectDelegate,
  required String csvOutputPath,
}) async {
  final Map<String, Map<int?, Map<String, _BenchStats>>> store = {
    'XNNPACK': {},
    'CPU': {},
    'NNAPI': {},
    'GPU': {},
  };

  final int cpuCount = Platform.numberOfProcessors;
  final List<int> clampedThreads =
      (threads.map((t) => (t.clamp(1, cpuCount) as int)).toSet().toList()
        ..sort());

  // XNNPACK sweep
  if (selectDelegate == "XNNPACK") {
    for (final t in clampedThreads) {
      final a = await _safeRunOne(
        modelPath: modelAPath,
        imageBytes: imageBytes,
        options: _optsXnn(t),
        runs: runs,
        warmup: warmup,
        debugTag: 'XNNPACK/$t $modelALabel',
      );
      final b = await _safeRunOne(
        modelPath: modelBPath,
        imageBytes: imageBytes,
        options: _optsXnn(t),
        runs: runs,
        warmup: warmup,
        debugTag: 'XNNPACK/$t $modelBLabel',
      );
      if (a != null || b != null) {
        store['XNNPACK']![t] = {
          if (a != null) modelALabel: a,
          if (b != null) modelBLabel: b,
        };
      }
    }
  }
  // CPU sweep
  else if (selectDelegate == "CPU") {
    for (final t in clampedThreads) {
      final a = await _safeRunOne(
        modelPath: modelAPath,
        imageBytes: imageBytes,
        options: _optsCpu(t),
        runs: runs,
        warmup: warmup,
        debugTag: 'CPU/$t $modelALabel',
      );
      final b = await _safeRunOne(
        modelPath: modelBPath,
        imageBytes: imageBytes,
        options: _optsCpu(t),
        runs: runs,
        warmup: warmup,
        debugTag: 'CPU/$t $modelBLabel',
      );
      if (a != null || b != null) {
        store['CPU']![t] = {
          if (a != null) modelALabel: a,
          if (b != null) modelBLabel: b,
        };
      }
    }
  }
  // NNAPI (best-effort)
  else if (selectDelegate == "NNAPI") {
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
  }

  if (selectDelegate == "GPU") {
    _BenchStats? aG;
    _BenchStats? bG;
    try {
      aG = await _safeRunOne(
        modelPath: modelAPath,
        imageBytes: imageBytes,
        options: _optsGpu(),
        runs: runs,
        warmup: warmup,
        debugTag: 'GPU $modelALabel',
      );
      bG = await _safeRunOne(
        modelPath: modelBPath,
        imageBytes: imageBytes,
        options: _optsGpu(),
        runs: runs,
        warmup: warmup,
        debugTag: 'GPU $modelBLabel',
      );
    } catch (_) {}
    if (aG != null || bG != null) {
      store['GPU']![null] = {
        if (aG != null) modelALabel: aG,
        if (bG != null) modelBLabel: bG,
      };
    }
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

  // CSV builder
  final csv = StringBuffer()
    ..writeln(
      "delegate,threads,model,avg_ms,min_ms,median_ms,p90_ms,max_ms,fps",
    );

  if (selectDelegate == "XNNPACK" || selectDelegate == "CPU") {
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

        // Markdown row
        buf.writeln("| **$delegate** | $t | ${cell(a)} | ${cell(b)} |");

        // CSV rows (one per model that exists)
        if (a != null) {
          csv.writeln(
            "$delegate,$t,$modelALabel,"
            "${a.avg},${a.minT},${a.median},${a.p90},${a.maxT},${a.fps}",
          );
        }
        if (b != null) {
          csv.writeln(
            "$delegate,$t,$modelBLabel,"
            "${b.avg},${b.minT},${b.median},${b.p90},${b.maxT},${b.fps}",
          );
        }
      }
    }

    if (selectDelegate == "XNNPACK") rows('XNNPACK');
    if (selectDelegate == "CPU") rows('CPU');
  } else if (selectDelegate == "NNAPI") {
    if (store['NNAPI']![null] != null && store['NNAPI']![null]!.isNotEmpty) {
      final aL = store['NNAPI']![null]?[modelALabel];
      final bL = store['NNAPI']![null]?[modelBLabel];
      buf.writeln("| **NNAPI** | – | ${cell(aL)} | ${cell(bL)} |");

      if (aL != null) {
        csv.writeln(
          "NNAPI,-,$modelALabel,"
          "${aL.avg},${aL.minT},${aL.median},${aL.p90},${aL.maxT},${aL.fps}",
        );
      }
      if (bL != null) {
        csv.writeln(
          "NNAPI,-,$modelBLabel,"
          "${bL.avg},${bL.minT},${bL.median},${bL.p90},${bL.maxT},${bL.fps}",
        );
      }
    }
  } else if (selectDelegate == "GPU") {
    if (store['GPU']![null] != null && store['GPU']![null]!.isNotEmpty) {
      final aL = store['GPU']![null]?[modelALabel];
      final bL = store['GPU']![null]?[modelBLabel];
      buf.writeln("| **GPU** | – | ${cell(aL)} | ${cell(bL)} |");

      if (aL != null) {
        csv.writeln(
          "GPU,-,$modelALabel,"
          "${aL.avg},${aL.minT},${aL.median},${aL.p90},${aL.maxT},${aL.fps}",
        );
      }
      if (bL != null) {
        csv.writeln(
          "GPU,-,$modelBLabel,"
          "${bL.avg},${bL.minT},${bL.median},${bL.p90},${bL.maxT},${bL.fps}",
        );
      }
    }
  }

  // final clampedThreadList = clampedThreads.join(', ');
  // buf
  //   ..writeln()
  //   ..writeln(
  //     "**Notes:** Runs=$runs, Warmup=$warmup. "
  //     "XNNPACK/CPU use threads=$clampedThreadList (clamped to $cpuCount cores). "
  //     "NNAPI ignores threads and is shown last.",
  //   );

  // Save CSV to disk
  try {
    final file = File(csvOutputPath);
    await file.writeAsString(csv.toString(), flush: true);
    print('[bench] CSV written to $csvOutputPath');
  } catch (e, st) {
    print('[bench] Failed to write CSV: $e\n$st');
  }

  return buf.toString();
}
