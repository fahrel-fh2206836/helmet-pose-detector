import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';

class CameraWidget extends StatefulWidget {
  final bool isActivated;
  const CameraWidget({super.key, required this.isActivated});

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
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (!widget.isActivated) return;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _controller?.dispose();
      _controller = null;
      _initializeControllerFuture = null;
    } else if (state == AppLifecycleState.resumed) {
      _checkAndInitialize(); // Re-initialize if still activated
    }
  }

  @override
  void didUpdateWidget(CameraWidget oldWidget) {
    super.didUpdateWidget(oldWidget);

    // Activate camera if parent just activated it
    if (widget.isActivated && !oldWidget.isActivated) {
      _checkAndInitialize();
    }

    // Clean up if parent just deactivated it
    if (!widget.isActivated && oldWidget.isActivated) {
      _controller?.dispose();
      _controller = null;
      _initializeControllerFuture = null;
    }
  }

  Future<void> _checkAndInitialize() async {
    if (!widget.isActivated) return;

    final status = await Permission.camera.status;

    if (status.isGranted) {
      await _initializeCamera();
    } else if (status.isDenied) {
      final result = await Permission.camera.request();
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
    try {
      final cameras = await availableCameras();
      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
      );

      _controller?.dispose(); // dispose old controller
      _controller = CameraController(frontCamera, ResolutionPreset.medium);
      _initializeControllerFuture = _controller!.initialize();

      setState(() {
        _permissionGranted = true;
        _permissionPermanentlyDenied = false;
      });
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.isActivated) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(8.0),
          child: Text(
            'Please press the activate button to activate the AI and Camera',
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

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

    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done &&
            _controller != null &&
            _controller!.value.isInitialized) {
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
