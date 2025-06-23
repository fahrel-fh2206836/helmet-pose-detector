import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;

class CameraImageUtils {
  /// Converts YUV420 CameraImage to RGB `img.Image`
  static Future<img.Image> convertCameraImage(CameraImage cameraImage) async {
    final int width = cameraImage.width;
    final int height = cameraImage.height;
    final img.Image image = img.Image(
      width: width,
      height: height,
    ); // Create RGB image

    final Uint8List yPlane = cameraImage.planes[0].bytes;
    final Uint8List uPlane = cameraImage.planes[1].bytes;
    final Uint8List vPlane = cameraImage.planes[2].bytes;

    final int uvRowStride = cameraImage.planes[1].bytesPerRow;
    final int uvPixelStride = cameraImage.planes[1].bytesPerPixel!;

    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        final int yIndex = h * width + w;
        final int uvIndex = uvPixelStride * (w ~/ 2) + uvRowStride * (h ~/ 2);

        final int y = yPlane[yIndex];
        final int u = uPlane[uvIndex];
        final int v = vPlane[uvIndex];

        int r = (y + 1.370705 * (v - 128)).round();
        int g = (y - 0.337633 * (u - 128) - 0.698001 * (v - 128)).round();
        int b = (y + 1.732446 * (u - 128)).round();

        image.setPixelRgb(
          w,
          h,
          r.clamp(0, 255),
          g.clamp(0, 255),
          b.clamp(0, 255),
        );
      }
    }

    return image;
  }

  /// Resizes and normalizes image to Float32List of shape [1, inputSize, inputSize, 3]
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
