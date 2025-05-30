import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

class CameraWidget extends StatefulWidget {
  const CameraWidget({super.key});

  @override
  State<CameraWidget> createState() => _CameraWidgetState();
}

class _CameraWidgetState extends State<CameraWidget>
    with WidgetsBindingObserver {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;
  bool _permissionGranted = false;
  bool _permissionPermanentlyDenied = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkAndInitialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    // _controller?.dispose();
    super.dispose();
  }

  // Called when app is resumed (after coming back from settings)
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _checkAndInitialize(); // re-check permission
    }
  }

  Future<void> _checkAndInitialize() async {
    var status = await Permission.camera.status;

    if (status.isGranted) {
      await _initializeCamera();
    } else if (status.isDenied) {
      var result = await Permission.camera.request();
      if (result.isGranted) {
        await _initializeCamera();
      } else if (result.isPermanentlyDenied) {
        setState(() => _permissionPermanentlyDenied = true);
      }
    } else if (status.isPermanentlyDenied) {
      setState(() => _permissionPermanentlyDenied = true);
    } else {
      setState(() {
        _permissionGranted = false;
      });
    }
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    final frontCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
    );

    _controller?.dispose(); // dispose old controller if any
    _controller = CameraController(frontCamera, ResolutionPreset.medium);
    _initializeControllerFuture = _controller!.initialize();

    setState(() {
      _permissionGranted = true;
      _permissionPermanentlyDenied = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_permissionPermanentlyDenied) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text(
              'Camera permission permanently denied.\nPlease enable it in app settings.',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () => openAppSettings(),
              child: const Text('Open Settings'),
            ),
          ],
        ),
      );
    }

    if (!_permissionGranted || _initializeControllerFuture == null) {
      return const Center(child: CircularProgressIndicator());
    }

    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          return AspectRatio(
            aspectRatio: _controller!.value.aspectRatio,
            child: CameraPreview(_controller!),
          );
        } else {
          return const Center(child: CircularProgressIndicator());
        }
      },
    );
  }
}
