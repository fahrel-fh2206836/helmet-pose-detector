// lib/services/permission_service.dart
import 'package:geolocator/geolocator.dart';
import 'package:permission_handler/permission_handler.dart';

class PermissionService {
  /// Ask for all core permissions your app needs.
  static Future<void> requestAllPermissions() async {
    await _handleLocation();
    await _handleNotifications();
    await _handleCamera();
  }

  static Future<void> _handleLocation() async {
    var status = await Geolocator.checkPermission();
    if (status == LocationPermission.denied) {
      await Geolocator.requestPermission();
    }
    // (Optionally) handle deniedForever via Geolocator.openAppSettings()
  }

  static Future<void> _handleNotifications() async {
    final status = await Permission.notification.status;
    if (status.isDenied) {
      await Permission.notification.request();
    }
  }

  static Future<void> _handleCamera() async {
    final status = await Permission.camera.status;
    if (status.isDenied) {
      await Permission.camera.request();
    }
  }

  static Future<bool> hasCameraPermission() async =>
      await Permission.camera.status.isGranted;
  static Future<bool> hasNotificationPermission() async =>
      await Permission.notification.status.isGranted;
  static Future<bool> hasLocationPermission() async =>
      await Geolocator.checkPermission() == LocationPermission.always ||
      await Geolocator.checkPermission() == LocationPermission.whileInUse;
}
