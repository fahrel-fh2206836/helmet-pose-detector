import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class HelmetDetector {
  final Interpreter _itp;

  HelmetDetector(this._itp);

  static Future<HelmetDetector> load({
    String assetPath = "assets/yolo11n_full_int8_320.tflite",
    int threads = 4,
  }) async {
    final options = InterpreterOptions()
      ..addDelegate(
        XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: threads)),
      )
      ..useNnApiForAndroid = false;

    final itp = await Interpreter.fromAsset(assetPath, options: options);
    return HelmetDetector(itp);
  }

  void close() => _itp.close();

  int get inputSize => _itp.getInputTensor(0).shape[1]; // [1,S,S,3]

  /// Letterbox with BLACK padding (matches your Roboflow setting)
  img.Image _letterboxBlack(img.Image src, int target) {
    final srcW = src.width, srcH = src.height;
    final scale = (target / srcW < target / srcH)
        ? (target / srcW)
        : (target / srcH);
    final newW = (srcW * scale).round();
    final newH = (srcH * scale).round();
    final resized = img.copyResize(src, width: newW, height: newH);

    final canvas = img.Image(width: target, height: target);
    img.fill(canvas, color: img.ColorRgb8(0, 0, 0));
    final dx = ((target - newW) / 2).round();
    final dy = ((target - newH) / 2).round();
    img.compositeImage(canvas, resized, dstX: dx, dstY: dy);
    return canvas;
  }

  /// Returns (helmetDetected, helmetConf) from NMS output [1,N,6] int8
  ({bool helmetDetected, double helmetConf}) runDetection(
    img.Image rgb, {
    double confThres = 0.5,
  }) {
    final S = inputSize;
    final padded = _letterboxBlack(rgb, S);

    final inTensor = _itp.getInputTensor(0);
    final s = inTensor.params.scale;
    final z = inTensor.params.zeroPoint;

    int q(double real01) => (real01 / s + z).round().clamp(-128, 127);

    // NHWC int8: [1][H][W][3]
    final hwc = List.generate(
      S,
      (_) => List.generate(
        S,
        (_) => List<int>.filled(3, 0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    for (int y = 0; y < S; y++) {
      for (int x = 0; x < S; x++) {
        final p = padded.getPixel(x, y);
        hwc[y][x][0] = q(p.r / 255.0);
        hwc[y][x][1] = q(p.g / 255.0);
        hwc[y][x][2] = q(p.b / 255.0);
      }
    }

    final input = [hwc];

    // Output [1,N,6] int8
    final outTensor = _itp.getOutputTensor(0);
    final outShape = outTensor.shape; // [1,N,6]
    final outScale = outTensor.params.scale;
    final outZero = outTensor.params.zeroPoint;

    final out = List.generate(
      outShape[0],
      (_) => List.generate(
        outShape[1],
        (_) => List<int>.filled(outShape[2], 0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    _itp.run(input, out);

    // Find max conf
    int maxQ = -128;
    for (int i = 0; i < out[0].length; i++) {
      final qConf = out[0][i][4];
      if (qConf > maxQ) maxQ = qConf;
    }

    final maxConf = (maxQ - outZero) * outScale;
    return (helmetDetected: maxConf >= confThres, helmetConf: maxConf);
  }
}
