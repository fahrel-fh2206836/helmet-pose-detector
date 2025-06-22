import 'dart:async';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:helmet_detector_app/widgets/camera_widget.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  bool _isActivated = false;
  Timer? _activationTimer;
  int _secondsElapsed = 0;
  double _currentSpeed = 0.0;

  @override
  void initState() {
    super.initState();
    _startTracking();
  }

  Future<void> _startTracking() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return;
    }
    if (permission == LocationPermission.deniedForever) return;

    Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 5,
      ),
    ).listen((Position position) {
      setState(() {
        _currentSpeed = (position.speed) * 3.6;
      });
    });
  }

  void _startTimer() {
    _secondsElapsed = 0;
    _activationTimer?.cancel();
    _activationTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      setState(() {
        _secondsElapsed++;
      });
    });
  }

  void _stopTimer() {
    _activationTimer?.cancel();
    _secondsElapsed = 0;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16.0),
              child: Text(
                'DriveSafe',
                style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
              ),
            ),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Stay Focused, Drive Safe',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: Colors.deepOrange,
                ),
              ),
            ),
            const SizedBox(height: 8),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              child: Text(
                'Please allow all the permissions that has been asked in order for the app to function properly',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 16),
              ),
            ),
            const SizedBox(height: 24),
            // Camera container
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 16.0),
              height: 250,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.green, width: 2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Center(child: CameraWidget(isActivated: _isActivated)),
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: 300,
              height: 50,
              child: ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isActivated
                      ? const Color(0xFF12EB66)
                      : Colors.grey,
                  foregroundColor: Colors.black,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 24,
                    vertical: 12,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                ),
                onPressed: () {
                  setState(() {
                    _isActivated = !_isActivated;
                  });
                  if (_isActivated) {
                    _startTimer();
                  } else {
                    _stopTimer();
                  }
                },
                icon: Icon(
                  !_isActivated ? Icons.power_settings_new : Icons.pause,
                ),
                label: Expanded(
                  child: Text(
                    _isActivated
                        ? 'Deactive AI & Camera'
                        : 'Activate AI & Camera',
                    style: TextStyle(fontSize: 18),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 30),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.timer, color: Colors.green),
                SizedBox(width: 20),
                Text(
                  _isActivated
                      ? 'Tracking for $_secondsElapsed seconds'
                      : 'Currently Not Tracking',
                  style: const TextStyle(fontSize: 16),
                ),
              ],
            ),
            const SizedBox(height: 30),
            _buildSpeedWidget(),
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: ''),
          BottomNavigationBarItem(icon: Icon(Icons.info), label: ''),
        ],
        onTap: (index) {},
      ),
    );
  }

  Widget _buildSpeedWidget() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        const Icon(Icons.motorcycle, color: Colors.green),
        const SizedBox(width: 8),
        Text(
          'Current Speed: ${_currentSpeed.toStringAsFixed(2)} km/h',
          style: const TextStyle(fontSize: 16),
        ),
      ],
    );
  }
}
