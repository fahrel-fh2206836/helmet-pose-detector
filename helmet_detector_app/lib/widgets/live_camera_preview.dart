import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

class LiveCameraView extends StatelessWidget {
  final CameraController controller;
  const LiveCameraView({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: controller.value.aspectRatio,
      child: CameraPreview(controller), // isolated from parent rebuilds
    );
  }
}