import 'dart:typed_data';
import 'package:camera/camera.dart';

class CameraImageUtils {
  static Future<Float32List> preprocessCameraImage(CameraImage image, {int targetSize = 416}) async {
  final int width = image.width;
  final int height = image.height;
  final int uvRowStride = image.planes[1].bytesPerRow;
  final int uvPixelStride = image.planes[1].bytesPerPixel!;

  final Uint8List yPlane = image.planes[0].bytes;
  final Uint8List uPlane = image.planes[1].bytes;
  final Uint8List vPlane = image.planes[2].bytes;

  final Float32List input = Float32List(targetSize * targetSize * 3);
  int pixelIndex = 0;

  final double scaleX = width / targetSize;
  final double scaleY = height / targetSize;

  for (int y = 0; y < targetSize; y++) {
    for (int x = 0; x < targetSize; x++) {
      final int srcX = (x * scaleX).floor();
      final int srcY = (y * scaleY).floor();

      final int yIndex = srcY * width + srcX;
      final int uvIndex = uvPixelStride * (srcX >> 1) + uvRowStride * (srcY >> 1);

      final int yp = yPlane[yIndex];
      final int up = uPlane[uvIndex];
      final int vp = vPlane[uvIndex];

      int r = yp + ((1436 * (vp - 128)) >> 10);
      int g = yp - ((46549 * (up - 128)) >> 17) - ((93604 * (vp - 128)) >> 17);
      int b = yp + ((1814 * (up - 128)) >> 10);

      input[pixelIndex++] = (r.clamp(0, 255)) / 255.0;
      input[pixelIndex++] = (g.clamp(0, 255)) / 255.0;
      input[pixelIndex++] = (b.clamp(0, 255)) / 255.0;
    }
  }

  return input;
}

}
