import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:helmet_detector_app/screens/main_screen.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
import 'package:helmet_detector_app/services/permission_service.dart';

// Uncomment to benchmark
// import 'package:helmet_detector_app/benchmark/benchmark_classifer_v1.dart';
// import 'package:helmet_detector_app/benchmark/benchmark_classifier_with_csv_v2.dart';
import 'package:helmet_detector_app/benchmark/benchmark_helmet_detection.dart';

// Note: App currently working only on Android phones.

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Benchmark (Uncomment when benchmarking and comment all he below code within the main function)
  // await runBenchmarkSuiteIsolatedWithCSV(
  //   deviceName: "S24+",
  //   modelALabel: "mobilenetv2_100_full_int8",
  //   modelAAsset: "assets/int8/mobilenetv2_100_full_int8.tflite",
  //   modelBLabel: "mobilenetv2_140_full_int8",
  //   modelBAsset: "assets/int8/mobilenetv2_140_full_int8.tflite",
  //   sampleImageAsset: "assets/sample.png",
  //   threads: const [4],
  //   runs: 100,
  //   warmup: 10,
  //   selectDelegate: "XNNPACK",
  // );

  await runYoloInt8NmsBenchmarkIsolatedWithCSV(
    deviceName: "S24+",
    modelLabel: "YOLOv11n INT8 640x640",
    modelAsset: "assets/int8/best_full_integer_quant.tflite",
    selectDelegate: "XNNPACK",
    threads: const [4],
    runs: 100,
    warmup: 10,
    imgSize: 320,
  );

  // Prevents Landscape orientation
  // await SystemChrome.setPreferredOrientations([
  //   DeviceOrientation.portraitUp, // Lock to upright portrait
  //   DeviceOrientation.portraitDown, // Allow upside-down portrait
  // ]);

  // // Requests all necesary permissions
  // await PermissionService.requestAllPermissions();

  // // Initializes notifcation services
  // await NotiService().initNotification();

  // runApp(const MyApp());
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
