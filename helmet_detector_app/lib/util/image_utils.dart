import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

class CameraImageUtils {
  static Future<img.Image> convertCameraImage(CameraImage cameraImage) async {
    final int width = cameraImage.width;
    final int height = cameraImage.height;

    final img.Image image = img.Image(width: width, height: height);

    final Uint8List y = cameraImage.planes[0].bytes;
    final Uint8List u = cameraImage.planes[1].bytes;
    final Uint8List v = cameraImage.planes[2].bytes;

    int uvRowStride = cameraImage.planes[1].bytesPerRow;
    int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        final int uvIndex = uvPixelStride * (w ~/ 2) + uvRowStride * (h ~/ 2);
        final int index = h * width + w;

        final int yp = y[index];
        final int up = u[uvIndex];
        final int vp = v[uvIndex];

        int r = (yp + 1.370705 * (vp - 128)).round();
        int g = (yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128)).round();
        int b = (yp + 1.732446 * (up - 128)).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        image.setPixelRgb(w, h, r, g, b);
      }
    }

    return image;
  }

  static Future<Float32List> preprocessImage(
    img.Image inputImage,
    int inputSize,
  ) async {
    final img.Image resized = img.copyResize(
      inputImage,
      width: inputSize,
      height: inputSize,
    );
    final Float32List input = Float32List(inputSize * inputSize * 3);

    int pixelIndex = 0;
    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);
        input[pixelIndex++] = pixel.r / 255.0;
        input[pixelIndex++] = pixel.g / 255.0;
        input[pixelIndex++] = pixel.b / 255.0;
      }
    }

    return input;
  }
}
