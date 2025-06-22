// import 'dart:typed_data';
// import 'package:camera/camera.dart';
// import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
// import 'package:image/image.dart' as img;

// /// Converts a CameraImage (YUV420) to a TensorImage suitable for TFLite model
// Future<TensorImage> convertCameraImage(CameraImage image) async {
//   final img.Image rgbImage = convertYUV420ToImage(image);

//   TensorImage tensorImage = TensorImage.fromImage(rgbImage);
//   tensorImage = ImageProcessorBuilder()
//       .add(ResizeOp(224, 224, ResizeMethod.BILINEAR)) // Adjust to your model input size
//       .build()
//       .process(tensorImage);

//   return tensorImage;
// }

// /// Converts CameraImage in YUV420 format to an RGB image
// img.Image convertYUV420ToImage(CameraImage image) {
//   final width = image.width;
//   final height = image.height;
//   final img.Image output = img.Image(width, height);

//   final uvRowStride = image.planes[1].bytesPerRow;
//   final uvPixelStride = image.planes[1].bytesPerPixel!;

//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       final uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
//       final yp = image.planes[0].bytes[y * image.planes[0].bytesPerRow + x];
//       final up = image.planes[1].bytes[uvIndex];
//       final vp = image.planes[2].bytes[uvIndex];
//       final r = (yp + 1.370705 * (vp - 128)).clamp(0, 255).toInt();
//       final g = (yp - 0.337633 * (up - 128) - 0.698001 * (vp - 128)).clamp(0, 255).toInt();
//       final b = (yp + 1.732446 * (up - 128)).clamp(0, 255).toInt();
//       output.setPixel(x, y, img.getColor(r, g, b));
//     }
//   }

//   return output;
// }
