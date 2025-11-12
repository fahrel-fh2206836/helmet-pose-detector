/* 
Packing for isolate (main isolate)
Convert CameraImage (YUV420) planes into a compact message with raw bytes,
strides, and layout flags. Uses TransferableTypedData for zero-copy transfer.
*/
import 'dart:isolate';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:helmet_detector_app/models/helmet_pose.dart';
import 'package:image/image.dart' as img;

Map<String, dynamic> packForIsolate(
  CameraImage image,
  bool isNCHW,
  HelmetPose model,
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
    'isNCHW': isNCHW, // controls tensor layout
    'buffer': ttd, // transferred bytes
    'model': model,
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
Future<({String label, double prob})> preprocessIsolate(
  Map<String, dynamic> msg,
) async {
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
  final bool isNCHW = msg['isNCHW'];
  final TransferableTypedData ttd = msg['buffer'] as TransferableTypedData;
  final HelmetPose model = msg['model'];

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

      int r = Y + ((1436 * (V - 128)) >> 10);
      int g = Y - ((46549 * (U - 128)) >> 17) - ((93604 * (V - 128)) >> 17);
      int b = Y + ((1814 * (U - 128)) >> 10);

      rgb.setPixelRgba(col, row, clamp(r), clamp(g), clamp(b), 255);
    }
  }

  // 2) resize directly to 224x224  (no extra crop step)
  const int size = 224;
  final im224 = img.copyResize(rgb, width: size, height: size);

  // 3) normalize with PyTorch mean/std and build nested list to match layout
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  Object inputNested;

  if (isNCHW) {
    // [1, 3, 224, 224]
    final c0 = List.generate(
      size,
      (_) => List<double>.filled(size, 0.0, growable: false),
      growable: false,
    );
    final c1 = List.generate(
      size,
      (_) => List<double>.filled(size, 0.0, growable: false),
      growable: false,
    );
    final c2 = List.generate(
      size,
      (_) => List<double>.filled(size, 0.0, growable: false),
      growable: false,
    );

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final p = im224.getPixel(x, y);
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
    // [1, 224, 224, 3]
    final hwc = List.generate(
      size,
      (_) => List.generate(
        size,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final p = im224.getPixel(x, y);
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

  return await model.runInference(inputNested);
}
