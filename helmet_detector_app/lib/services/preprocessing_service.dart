/* 
Packing for isolate (main isolate)
Convert CameraImage (YUV420) planes into a compact message with raw bytes,
strides, and layout flags. Uses TransferableTypedData for zero-copy transfer.
*/
import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/models/helmet_detector.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:image/image.dart' as img;

Map<String, dynamic> packForIsolate(
  CameraImage image,
  HelmetPose pose,
  HelmetDetector det,
) {
  final w = image.width;
  final h = image.height;

  // YUV420: planes[0]=Y, [1]=U, [2]=V
  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];

  // Concatenate Y,U,V for fast transfer
  final y = yPlane.bytes, u = uPlane.bytes, v = vPlane.bytes;
  final all = Uint8List(y.length + u.length + v.length)
    ..setRange(0, y.length, y)
    ..setRange(y.length, y.length + u.length, u)
    ..setRange(y.length + u.length, y.length + u.length + v.length, v);

  // Wrap in TTD to avoid copying when sending to another isolate
  final ttd = TransferableTypedData.fromList([all]);

  return {
    'w': w,
    'h': h,
    'yStride': yPlane.bytesPerRow, // row stride for Y
    'uStride': uPlane.bytesPerRow, // row stride for U
    'vStride': vPlane.bytesPerRow, // row stride for V
    'uPixStride': uPlane.bytesPerPixel ?? 1, // chroma pixel stride (often 2)
    'vPixStride': vPlane.bytesPerPixel ?? 1, // chroma pixel stride
    'yLen': y.length, // split points inside `all`
    'uLen': u.length,
    'buffer': ttd, // transferred bytes
    'pose': pose,
    'det': det,
  };
}

/* 
Heavy preprocessing (background isolate)
Converts YUV420 → RGB, resizes, crops, normalizes, and packs into nested
Dart lists that match the interpreter's expected input shape.
Returns `Object inputNested`:
- NCHW: [[c0(HxW), c1(HxW), c2(HxW)]]    // shape [1,3,300,300]
- NHWC: [ HxW x [r,g,b] ]   
*/
Future<({String label, double prob, bool helmetDetected, double helmetConf})>
preprocessAndInferenceIsolate(Map<String, dynamic> msg) async {
  // Unpack metadata
  final int srcW = msg['w'];
  final int srcH = msg['h'];
  final int yStride = msg['yStride'];
  final int uStride = msg['uStride'];
  final int vStride = msg['vStride'];
  final int uPixStride = msg['uPixStride'];
  final int vPixStride = msg['vPixStride'];
  final int yLen = msg['yLen'];
  final int uLen = msg['uLen'];
  final TransferableTypedData ttd = msg['buffer'] as TransferableTypedData;
  final HelmetPose pose = msg['pose'];
  final HelmetDetector det = msg['det'];

  // Read model input layout only once (controls how we pack tensors)
  final isNCHW = pose.isNCHW;
  final height = pose.modelHeight;
  final width = pose.modelWidth;

  // Recover concatenated YUV bytes and split them
  final Uint8List all = ttd.materialize().asUint8List();
  final Uint8List y = all.sublist(0, yLen);
  final Uint8List u = all.sublist(yLen, yLen + uLen);
  final Uint8List v = all.sublist(yLen + uLen);

  // 1) YUV420 -> RGB img.Image
  final rgb = img.Image(width: srcW, height: srcH);
  int clamp(int v) => v < 0 ? 0 : (v > 255 ? 255 : v);
  for (int row = 0; row < srcH; row++) {
    final yRow = row * yStride;
    final uRow = (row >> 1) * uStride;
    final vRow = (row >> 1) * vStride;
    for (int col = 0; col < srcW; col++) {
      final yIdx = yRow + col;
      final uIdx = uRow + (col >> 1) * uPixStride;
      final vIdx = vRow + (col >> 1) * vPixStride;
      final Y = y[yIdx];
      final U = u[uIdx];
      final V = v[vIdx];

      // Integer YUV→RGB conversion (BT.601-ish), bit shifts for speed
      int r = Y + ((1436 * (V - 128)) >> 10);
      int g = Y - ((46549 * (U - 128)) >> 17) - ((93604 * (V - 128)) >> 17);
      int b = Y + ((1814 * (U - 128)) >> 10);

      rgb.setPixelRgba(col, row, clamp(r), clamp(g), clamp(b), 255);
    }
  }

  //  YOLO helmet detect FIRST
  final detRes = det.runDetection(rgb, confThres: .55);

  // If no helmet → skip pose inference (saves time)
  if (!detRes.helmetDetected) {
    return (
      label: '-',
      prob: 0.0,
      helmetDetected: false,
      helmetConf: detRes.helmetConf,
    );
  }

  final img.Image im;
  im = img.copyResize(rgb, width: width, height: height);

  // normalize with PyTorch mean/std and build nested list to match layout
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  Object inputNested;

  if (isNCHW) {
    // [1, 3, 300, 300]
    // Allocate 3 separate 2D arrays, one per channel (R, G, B).
    final c0 = List.generate(
      height,
      (_) => List<double>.filled(width, 0.0, growable: false),
      growable: false,
    );
    final c1 = List.generate(
      height,
      (_) => List<double>.filled(width, 0.0, growable: false),
      growable: false,
    );
    final c2 = List.generate(
      height,
      (_) => List<double>.filled(width, 0.0, growable: false),
      growable: false,
    );

    // Fill each channel with normalized float values.
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final p = im.getPixel(x, y);
        final r = ((p.r) / 255.0 - mean[0]) / std[0];
        final g = ((p.g) / 255.0 - mean[1]) / std[1];
        final b = ((p.b) / 255.0 - mean[2]) / std[2];
        c0[y][x] = r;
        c1[y][x] = g;
        c2[y][x] = b;
      }
    }
    inputNested = [
      [c0, c1, c2],
    ];
  } else {
    // [1, 300, 300, 3]
    // Allocate H×W×3 (per pixel: [R,G,B]).
    final hwc = List.generate(
      height,
      (_) => List.generate(
        width,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    // Fill HWC with normalized float values.
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final p = im.getPixel(x, y);
        final r = ((p.r) / 255.0 - mean[0]) / std[0];
        final g = ((p.g) / 255.0 - mean[1]) / std[1];
        final b = ((p.b) / 255.0 - mean[2]) / std[2];
        hwc[y][x][0] = r;
        hwc[y][x][1] = g;
        hwc[y][x][2] = b;
      }
    }
    inputNested = [hwc];
  }

  // Direct run on tensor (no decode/encode)
  final poseRes = await pose.runInference(inputNested);

  return (
    label: poseRes.label,
    prob: poseRes.prob,
    helmetDetected: detRes.helmetDetected,
    helmetConf: detRes.helmetConf,
  );
}
