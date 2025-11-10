import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:helmet_detector_app/screens/main_screen.dart';
import 'package:helmet_detector_app/services/noti_service.dart';
import 'package:helmet_detector_app/services/permission_service.dart';

// Uncomment to benchmark
import 'package:helmet_detector_app/benchmark/benchmark_v2.dart';

// Note: App currently working only on Android phones.

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Benchmark (Uncomment when benchmarking and comment all he below code within the main function)
  await runBenchmarkSuiteIsolated(
    deviceName: "Samsung Galaxy S24+",
    modelALabel: "FP32",
    modelAAsset: "assets/mobilenet_fp32.tflite",
    modelBLabel: "FP32 I/O + FP16 internal",
    modelBAsset: "assets/mobilenet_int8_fp32io.tflite",
    sampleImageAsset: "assets/sample.png",
    threads: const [1, 2, 4, 6, 8, 10],
    runs: 30,
    warmup: 5,
    selectDelegate: "NNAPI",
  );

  // // Prevents Landscape orientation
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
