// Prepares the interpreter + model and benchmarks delegate latency (inference only)
import 'package:flutter/services.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

Future<void> benchmarkHelmetPose({
  required String name,
  required String assetPath,
  required Uint8List sampleImageBytes,
  InterpreterOptions? options,
  int runs = 20,
  int warmup = 5,
}) async {
  print('🔹 Benchmarking mode: $name');

  // Build interpreter with chosen delegate
  final itp = await Interpreter.fromAsset(assetPath, options: options);
  // Optional but good practice: force tensor allocation & shapes now
  itp.allocateTensors();

  final pose = HelmetPose(itp);

  // Derive input shape/format from the interpreter (or from HelmetPose if you store it there)
  // Example assumes your model takes [1, H, W, 3] or [1, 3, H, W].
  final inShape = itp.getInputTensor(0).shape;
  late final bool isNCHW;
  late final int H, W;

  if (inShape.length == 4 && inShape[1] == 3) {
    // [1, 3, H, W]
    isNCHW = true;
    H = inShape[2];
    W = inShape[3];
  } else if (inShape.length == 4 && inShape[3] == 3) {
    // [1, H, W, 3]
    isNCHW = false;
    H = inShape[1];
    W = inShape[2];
  } else {
    throw StateError('Unsupported input shape: $inShape');
  }

  // ---- Preprocess ONCE (excluded from timing)
  final inputNested = buildInputFromBytes(
    bytes: sampleImageBytes,
    isNCHW: isNCHW,
    H: H,
    W: W,
  );

  // ---- Warmup (excluded from timing)
  for (int i = 0; i < warmup; i++) {
    await pose.runInference(inputNested);
  }

  // ---- Timed runs
  final times = <double>[];
  for (int i = 0; i < runs; i++) {
    final sw = Stopwatch()..start();
    await pose.runInference(inputNested);
    sw.stop();
    times.add(sw.elapsedMicroseconds / 1000.0);
  }

  // ---- Metrics
  times.sort();
  final avg = times.reduce((a, b) => a + b) / times.length;
  final minT = times.first;
  final maxT = times.last;
  final median = times.length.isOdd
      ? times[times.length ~/ 2]
      : (times[times.length ~/ 2 - 1] + times[times.length ~/ 2]) / 2.0;
  double p(double q) => times[((q * (times.length - 1))).round()];
  final p90 = p(0.90);

  final fps = 1000.0 / avg;

  print(
    '✅ $name | avg=${avg.toStringAsFixed(2)} ms '
    '(min=${minT.toStringAsFixed(1)}, median=${median.toStringAsFixed(1)}, p90=${p90.toStringAsFixed(1)}, max=${maxT.toStringAsFixed(1)}) '
    '| FPS≈${fps.toStringAsFixed(1)}\n',
  );

  pose.close();
}

// Build the nested Dart list in exactly the shape the model expects.
Object buildInputFromBytes({
  required Uint8List bytes,
  required bool isNCHW,
  required int H,
  required int W,
}) {
  // Decode bytes → resize → center-crop to HxW
  img.Image? im = img.decodeImage(bytes);
  if (im == null) throw StateError("Unsupported image data.");
  // Your model uses 300x300 but you resize to 320 then crop center
  im = img.copyResize(im, width: 320, height: 320);
  final off = (320 - W) ~/ 2;
  im = img.copyCrop(im, x: off, y: off, width: W, height: H);

  // EfficientNet-B3 / PyTorch normalization
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  if (isNCHW) {
    // Shape: [1, 3, H, W]
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
    return [
      [c0, c1, c2],
    ];
  } else {
    // Shape: [1, H, W, 3]
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
    return [hwc];
  }
}

Future<void> testDelegates() async {
  final bytes = await rootBundle.load('assets/sample.png');
  final imageBytes = bytes.buffer.asUint8List();
  // const modelPath = 'assets/helmet_pose_fp32io_fp16.tflite';
  const modelPath = 'assets/helmet_pose_fp32io_fp16_nhwc.tflite';

  // final xnnpackOpts = InterpreterOptions()
  //   ..threads = 4
  //   ..addDelegate(XNNPackDelegate());
  // await benchmarkHelmetPose(
  //   name: 'XNNPACK',
  //   assetPath: modelPath,
  //   sampleImageBytes: imageBytes,
  //   options: xnnpackOpts,
  //   runs: 30,
  //   warmup: 5,
  // );

  // final nnapiOpts = InterpreterOptions()
  //   ..threads = 4
  //   ..useNnApiForAndroid = true;
  // await benchmarkHelmetPose(
  //   name: 'NNAPI delegate',
  //   assetPath: modelPath,
  //   sampleImageBytes: imageBytes,
  //   options: nnapiOpts,
  //   runs: 30,
  //   warmup: 5,
  // );

  // final defaultOpts = InterpreterOptions()..threads = 4;
  // await benchmarkHelmetPose(
  //   name: 'Default',
  //   assetPath: modelPath,
  //   sampleImageBytes: imageBytes,
  //   options: defaultOpts,
  //   runs: 30,
  //   warmup: 5,
  // );
}
