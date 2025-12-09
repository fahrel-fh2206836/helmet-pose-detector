import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:helmet_detector_app/screens/main_screen.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
import 'package:helmet_detector_app/services/permission_service.dart';

// Uncomment to benchmark
// import 'package:helmet_detector_app/benchmark/benchmark_v2.dart';
// import 'package:helmet_detector_app/benchmark/benchmark_v3_with_csv.dart';

// Note: App currently working only on Android phones.

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Benchmark (Uncomment when benchmarking and comment all he below code within the main function)
  // await runBenchmarkSuiteIsolatedWithCSV(
  //   deviceName: "A52",
  //   modelALabel: "ghostnet_100",
  //   modelAAsset: "assets/ghostnet_100_fp32io_fp16.tflite",
  //   modelBLabel: "mobilenetv3_small_100",
  //   modelBAsset: "assets/mobilenetv3_small_100_fp32io_fp16.tflite",
  //   sampleImageAsset: "assets/sample.png",
  //   threads: const [3],
  //   runs: 100,
  //   warmup: 10,
  //   selectDelegate: "XNNPACK",
  // );

  // Prevents Landscape orientation
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp, // Lock to upright portrait
    DeviceOrientation.portraitDown, // Allow upside-down portrait
  ]);

  // Requests all necesary permissions
  await PermissionService.requestAllPermissions();

  // Initializes notifcation services
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
