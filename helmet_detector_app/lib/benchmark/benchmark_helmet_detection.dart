// benchmark_isolate_yolo_int8_nms_single.dart
//
// Benchmarks ONE Ultralytics-exported YOLOv8 INT8 TFLite model that already includes NMS
// and returns a SINGLE output tensor shaped [1, 300, 6] (int8), e.g. output name "PartitionedCall:0".
//
// Matches your Roboflow preprocessing: Resize "Fit (black edges)" => letterbox with BLACK padding.
// Matches your Colab tensor details:
//   Input : [1,640,640,3] int8  quant(scale≈0.0039215689, zeroPoint=-128)
//   Output: [1,300,6]     int8  quant(scale≈0.0040839640, zeroPoint=-120)
//
// Measures end-to-end model latency INCLUDING NMS (since NMS is baked in).

import 'dart:async';
import 'dart:isolate';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// -------------------- Public entrypoint (call this from UI) --------------------
Future<void> runYoloInt8NmsBenchmarkIsolatedWithCSV({
  required String deviceName,
  required String modelLabel, // e.g. "YOLOv8n INT8 (NMS)"
  required String modelAsset, // e.g. "assets/best_full_integer_quant.tflite"
  String sampleImageAsset = 'assets/sample.png',
  List<int> threads = const [1, 2, 3, 4, 6, 8],
  int runs = 100,
  int warmup = 10,

  /// "XNNPACK" | "CPU" | "NNAPI" | "GPU"
  String selectDelegate = 'XNNPACK',

  /// YOLO input size
  int imgSize = 640,
}) async {
  // Prepare assets on MAIN isolate
  final modelPath = await _copyAssetToTemp(modelAsset);
  final imageBytes = (await rootBundle.load(
    sampleImageAsset,
  )).buffer.asUint8List();

  // Decide where to save CSV
  final docsDir = await getApplicationDocumentsDirectory();
  final csvPath =
      '${docsDir.path}/benchmark_yolo_${deviceName}_$selectDelegate.csv';

  // Run in background isolate
  final markdown = await Isolate.run(() async {
    return await _workerBenchmarkSingleModel(
      deviceName: deviceName,
      modelLabel: modelLabel,
      modelPath: modelPath,
      imageBytes: imageBytes,
      threads: threads,
      runs: runs,
      warmup: warmup,
      selectDelegate: selectDelegate,
      csvOutputPath: csvPath,
      imgSize: imgSize,
    );
  });

  print(markdown);
  print("YOLO INT8 (NMS) Benchmark Completed for $selectDelegate");
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

/// Roboflow: "Resize: Fit (black edges)" => letterbox with BLACK padding.
img.Image _letterboxBlack(img.Image src, int target) {
  final srcW = src.width;
  final srcH = src.height;

  final scale = (target / srcW < target / srcH)
      ? (target / srcW)
      : (target / srcH);
  final newW = (srcW * scale).round();
  final newH = (srcH * scale).round();

  final resized = img.copyResize(src, width: newW, height: newH);

  final canvas = img.Image(width: target, height: target);
  img.fill(canvas, color: img.ColorRgb8(0, 0, 0)); // black edges
  final dx = ((target - newW) / 2).round();
  final dy = ((target - newH) / 2).round();

  img.compositeImage(canvas, resized, dstX: dx, dstY: dy);
  return canvas;
}

/// Build YOLO input for your exact model:
/// - NHWC [1,640,640,3]
/// - int8 quant with model's scale/zeroPoint
Object _buildYoloInt8InputNHWC({
  required Uint8List bytes,
  required Interpreter interpreter,
  required int imgSize,
}) {
  img.Image? im = img.decodeImage(bytes);
  if (im == null) throw StateError("Unsupported image data.");

  final padded = _letterboxBlack(im, imgSize);

  final inTensor = interpreter.getInputTensor(0);
  final inShape = inTensor.shape; // expected [1,640,640,3]
  final inType = inTensor.type;

  if (inShape.length != 4 ||
      inShape[0] != 1 ||
      inShape[1] != imgSize ||
      inShape[2] != imgSize ||
      inShape[3] != 3) {
    throw StateError(
      "Unexpected input shape: $inShape (expected [1,$imgSize,$imgSize,3])",
    );
  }

  if (inType != TensorType.int8) {
    throw StateError("Expected int8 input, got $inType");
  }

  final s = inTensor.params.scale;
  final z = inTensor.params.zeroPoint;

  int q(double real01) => (real01 / s + z).round().clamp(-128, 127);

  // NHWC: [1][H][W][3]
  final hwc = List.generate(
    imgSize,
    (_) => List.generate(imgSize, (_) => List<int>.filled(3, 0)),
  );

  for (int y = 0; y < imgSize; y++) {
    for (int x = 0; x < imgSize; x++) {
      final p = padded.getPixel(x, y);
      hwc[y][x][0] = q(p.r / 255.0);
      hwc[y][x][1] = q(p.g / 255.0);
      hwc[y][x][2] = q(p.b / 255.0);
    }
  }

  return [hwc];
}

/// Allocate output container for your model's single output:
/// Output: [1,300,6] int8
List<List<List<int>>> _allocateYoloNmsOutput(Interpreter interpreter) {
  final outTensor = interpreter.getOutputTensor(0);
  final shape = outTensor.shape; // expected [1,300,6]
  final type = outTensor.type;

  if (shape.length != 3 || shape[0] != 1 || shape[2] != 6) {
    throw StateError("Unexpected output shape: $shape (expected [1,N,6])");
  }
  if (type != TensorType.int8) {
    throw StateError("Expected int8 output, got $type");
  }

  return List.generate(
    shape[0],
    (_) => List.generate(shape[1], (_) => List<int>.filled(shape[2], 0)),
  );
}

/// Run inference once (no decoding; pure latency).
void _runYoloOnce({
  required Object input,
  required Interpreter interpreter,
  required List<List<List<int>>> output,
}) {
  interpreter.run(input, output);
}

/// Bench one config (delegate/options).
Future<_BenchStats> _runOne({
  required String modelPath,
  required Uint8List imageBytes,
  required InterpreterOptions options,
  required int runs,
  required int warmup,
  required int imgSize,
}) async {
  final itp = Interpreter.fromFile(File(modelPath), options: options);
  itp.allocateTensors();

  // Build input ONCE (outside timing loop)
  final input = _buildYoloInt8InputNHWC(
    bytes: imageBytes,
    interpreter: itp,
    imgSize: imgSize,
  );

  // Allocate output ONCE
  final output = _allocateYoloNmsOutput(itp);

  // Warmup
  for (int i = 0; i < warmup; i++) {
    _runYoloOnce(input: input, interpreter: itp, output: output);
  }

  // One-time sanity check (OUTSIDE timed loop)
  _sanityCheckOnce(interpreter: itp, outputQ: output, confThres: 0.5, topK: 3);

  // Timed runs
  final times = <double>[];
  for (int i = 0; i < runs; i++) {
    final sw = Stopwatch()..start();
    _runYoloOnce(input: input, interpreter: itp, output: output);
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

/// Safe wrapper
Future<_BenchStats?> _safeRunOne({
  required String modelPath,
  required Uint8List imageBytes,
  required InterpreterOptions options,
  required int runs,
  required int warmup,
  required int imgSize,
  String debugTag = '',
}) async {
  try {
    return await _runOne(
      modelPath: modelPath,
      imageBytes: imageBytes,
      options: options,
      runs: runs,
      warmup: warmup,
      imgSize: imgSize,
    );
  } catch (e, st) {
    print('[bench-yolo] $debugTag failed: $e\n$st');
    return null;
  }
}

/// -------------------- Main benchmark suite (single model) --------------------
Future<String> _workerBenchmarkSingleModel({
  required String deviceName,
  required String modelLabel,
  required String modelPath,
  required Uint8List imageBytes,
  required List<int> threads,
  required int runs,
  required int warmup,
  required String selectDelegate,
  required String csvOutputPath,
  required int imgSize,
}) async {
  final Map<String, Map<int?, _BenchStats>> store = {
    'XNNPACK': {},
    'CPU': {},
    'NNAPI': {},
    'GPU': {},
  };

  final int cpuCount = Platform.numberOfProcessors;
  final List<int> clampedThreads =
      (threads.map((t) => (t.clamp(1, cpuCount) as int)).toSet().toList()
        ..sort());

  if (selectDelegate == "XNNPACK") {
    for (final t in clampedThreads) {
      final s = await _safeRunOne(
        modelPath: modelPath,
        imageBytes: imageBytes,
        options: _optsXnn(t),
        runs: runs,
        warmup: warmup,
        imgSize: imgSize,
        debugTag: 'XNNPACK/$t $modelLabel',
      );
      if (s != null) store['XNNPACK']![t] = s;
    }
  } else if (selectDelegate == "CPU") {
    for (final t in clampedThreads) {
      final s = await _safeRunOne(
        modelPath: modelPath,
        imageBytes: imageBytes,
        options: _optsCpu(t),
        runs: runs,
        warmup: warmup,
        imgSize: imgSize,
        debugTag: 'CPU/$t $modelLabel',
      );
      if (s != null) store['CPU']![t] = s;
    }
  } else if (selectDelegate == "NNAPI") {
    final s = await _safeRunOne(
      modelPath: modelPath,
      imageBytes: imageBytes,
      options: _optsNnapi(),
      runs: runs,
      warmup: warmup,
      imgSize: imgSize,
      debugTag: 'NNAPI $modelLabel',
    );
    if (s != null) store['NNAPI']![null] = s;
  } else if (selectDelegate == "GPU") {
    final s = await _safeRunOne(
      modelPath: modelPath,
      imageBytes: imageBytes,
      options: _optsGpu(),
      runs: runs,
      warmup: warmup,
      imgSize: imgSize,
      debugTag: 'GPU $modelLabel',
    );
    if (s != null) store['GPU']![null] = s;
  } else {
    throw StateError(
      'Unknown delegate "$selectDelegate" (use XNNPACK/CPU/NNAPI/GPU)',
    );
  }

  String cell(_BenchStats? s) => s == null ? "N/A" : s.toCell();

  final buf = StringBuffer()
    ..writeln("### 📊 YOLOv8 INT8 (NMS) TFLite Benchmark — $deviceName")
    ..writeln()
    ..writeln(
      "| **Delegate** | **Threads** | **$modelLabel**<br>Avg / Min / Median / P90 / Max / FPS |",
    )
    ..writeln(
      "|:-------------|:-----------:|:----------------------------------------------------------|",
    );

  // CSV builder
  final csv = StringBuffer()
    ..writeln(
      "delegate,threads,model,avg_ms,min_ms,median_ms,p90_ms,max_ms,fps",
    );

  void addRow(String delegate, int? t, _BenchStats? s) {
    final threadCell = t == null ? "–" : "$t";
    buf.writeln("| **$delegate** | $threadCell | ${cell(s)} |");

    if (s != null) {
      csv.writeln(
        "$delegate,${t ?? "-"},$modelLabel,"
        "${s.avg},${s.minT},${s.median},${s.p90},${s.maxT},${s.fps}",
      );
    }
  }

  if (selectDelegate == "XNNPACK" || selectDelegate == "CPU") {
    final entries = store[selectDelegate]!;
    final keys = entries.keys.where((k) => k != null).cast<int>().toList()
      ..sort();
    for (final t in keys) {
      addRow(selectDelegate, t, entries[t]);
    }
  } else {
    addRow(selectDelegate, null, store[selectDelegate]![null]);
  }

  // Save CSV to disk
  try {
    final file = File(csvOutputPath);
    await file.writeAsString(csv.toString(), flush: true);
    print('[bench-yolo] CSV written to $csvOutputPath');
  } catch (e, st) {
    print('[bench-yolo] Failed to write CSV: $e\n$st');
  }

  return buf.toString();
}

void _sanityCheckOnce({
  required Interpreter interpreter,
  required List<List<List<int>>> outputQ, // int8 output [1][300][6]
  double confThres = 0.5,
  int topK = 3,
}) {
  final outTensor = interpreter.getOutputTensor(0);
  final outScale = outTensor.params.scale;
  final outZero = outTensor.params.zeroPoint;

  // Quantized threshold for conf
  final qThres = (confThres / outScale + outZero).round().clamp(-128, 127);

  int count = 0;
  int maxQ = -128;
  int maxIdx = -1;

  // outputQ[0][i][4] is conf (int8)
  for (int i = 0; i < outputQ[0].length; i++) {
    final qConf = outputQ[0][i][4];
    if (qConf > qThres) count++;
    if (qConf > maxQ) {
      maxQ = qConf;
      maxIdx = i;
    }
  }

  final maxConf = (maxQ - outZero) * outScale;
  print(
    '[sanity] dets(conf>${confThres.toStringAsFixed(2)}): $count'
    ' | maxConf=${maxConf.toStringAsFixed(3)} (row=$maxIdx)',
  );

  // Print a few top rows (by quantized conf) for quick inspection (optional)
  // This is outside timing so it's fine.
  final idxs = List<int>.generate(outputQ[0].length, (i) => i);
  idxs.sort((a, b) => outputQ[0][b][4].compareTo(outputQ[0][a][4]));

  print('[sanity] top $topK rows [x1,y1,x2,y2,conf,cls] (DEQUANTIZED):');
  for (int k = 0; k < topK && k < idxs.length; k++) {
    final i = idxs[k];
    final rowQ = outputQ[0][i];

    // dequantize just this row
    double dq(int q) => (q - outZero) * outScale;

    final x1 = dq(rowQ[0]);
    final y1 = dq(rowQ[1]);
    final x2 = dq(rowQ[2]);
    final y2 = dq(rowQ[3]);
    final conf = dq(rowQ[4]);
    final cls = dq(rowQ[5]);

    print('  $k) [$x1, $y1, $x2, $y2, $conf, $cls]');
  }

  // Helpful reminder (since your Colab confirmed normalized coords)
  print(
    '[sanity] note: coords appear normalized → multiply by 640 for pixel space.',
  );
}
