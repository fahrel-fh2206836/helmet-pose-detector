import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';

class SpeedWidget extends StatefulWidget {
  const SpeedWidget({super.key});

  @override
  State<SpeedWidget> createState() => _SpeedWidgetState();
}

class _SpeedWidgetState extends State<SpeedWidget> {
  double _currentSpeed = 0.0;

  @override
  void initState() {
    super.initState();
    _startTracking();
  }

  Future<void> _startTracking() async {
    bool serviceEnabled;
    LocationPermission permission;

    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      return;
    }

    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        return;
      }
    }

    if (permission == LocationPermission.deniedForever) {
      return;
    }

    Geolocator.getPositionStream(
      locationSettings: const LocationSettings(
        accuracy: LocationAccuracy.high,
        distanceFilter: 5,
      ),
    ).listen((Position position) {
      setState(() {
        _currentSpeed = (position.speed) * 3.6; // m/s to km/h
      });
    });
  }

  @override
  Widget build(BuildContext context) {
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
