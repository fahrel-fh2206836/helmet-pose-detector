import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:helmet_detector_app/screens/main_screen.dart';
import 'package:helmet_detector_app/util/noti_service.dart';
import 'package:permission_handler/permission_handler.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await _handleAllPermissions();

  runApp(const MyApp());
}

Future<void> _handleAllPermissions() async {
  // 📍 LOCATION
  LocationPermission locationPermission = await Geolocator.checkPermission();
  if (locationPermission == LocationPermission.denied) {
    locationPermission = await Geolocator.requestPermission();
  }

  // 🔔 NOTIFICATION
  var notificationStatus = await Permission.notification.status;
  if (notificationStatus.isDenied) {
    final result = await Permission.notification.request();
    if (result.isGranted) {
      await NotiService().initNotification();
    }
  } else if (notificationStatus.isGranted) {
    await NotiService().initNotification();
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'RideSafe',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.greenAccent),
      ),
      home: const MainScreen(),
    );
  }
}
