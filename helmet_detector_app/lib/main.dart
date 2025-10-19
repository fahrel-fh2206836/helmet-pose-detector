import 'package:flutter/material.dart';
import 'package:helmet_detector_app/screens/main_screen.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
import 'package:helmet_detector_app/services/permission_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Centralized permission requests
  await PermissionService.requestAllPermissions();

  // Safe to init notifications (permission was requested already).
  await NotiService().initNotification();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

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
