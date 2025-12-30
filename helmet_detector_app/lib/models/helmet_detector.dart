import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class HelmetDetector {
  final Interpreter _itp;
  int helmetClass = 0; // or 1

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

  /// Returns (helmetDetected, helmetConf) from NMS output [1,N,6] int8
  ({bool helmetDetected, double helmetConf}) runDetection(
    img.Image rgb, {
    double confThres = 0.5,
  }) {
    final S = inputSize;
    final padded = img.copyResize(rgb, width: S, height: S);

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

    // NMS layout: [x1, y1, x2, y2, conf, cls]
    const int confIdx = 4;
    const int clsIdx = 5;

    double deq(int q) => (q - outZero) * outScale;

    double bestHelmet = 0.0;
    bool helmetDetected = false;

    for (int i = 0; i < out[0].length; i++) {
      final row = out[0][i];

      final conf = deq(row[confIdx]);
      if (conf < confThres) continue;

      // --- decode class id robustly (some NMS int8 exports store cls raw, others quantized) ---
      final int clsRaw = row[clsIdx];
      final int clsDeq = deq(row[clsIdx]).round();

      // Prefer a decoded value that falls into valid class range [0..1]
      final int cls = (clsDeq == 0 || clsDeq == 1)
          ? clsDeq
          : ((clsRaw == 0 || clsRaw == 1) ? clsRaw : clsDeq);

      if (cls == helmetClass) {
        helmetDetected = true;
        if (conf > bestHelmet) bestHelmet = conf;
        // optional: break if you only need a boolean and don't care about max conf
        // break;
      }
    }

    return (helmetDetected: helmetDetected, helmetConf: bestHelmet);
  }
}
