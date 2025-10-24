// lib/services/speed_service.dart
import 'dart:async';
import 'package:geolocator/geolocator.dart';

/// Emits current ground speed in km/h, with simple smoothing.
/// Call start() once, listen to [speedStream], and stop()/dispose() when done.
class SpeedService {
  final _controller = StreamController<double>.broadcast();
  Stream<double> get speedStream => _controller.stream;

  // public, always holds last emitted speed (km/h)
  double latestKmh = 0.0;

  StreamSubscription<Position>? _sub;

  // Basic smoothing over the last N samples
  final int _window;
  final List<double> _buf = [];

  // Tweak location settings as needed
  final LocationSettings _settings;

  SpeedService({int smoothingWindow = 4, LocationSettings? settings})
    : _window = smoothingWindow.clamp(1, 20),
      _settings =
          settings ??
          const LocationSettings(
            accuracy: LocationAccuracy.high,
            distanceFilter: 5, // meters
          );

  Future<void> start() async {
    // Ensure location services & permission are OK (optional; can be handled elsewhere)
    final serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      // You can surface a UI prompt elsewhere; we keep service silent here
    }

    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied) {
      return;
    }

    _sub ??= Geolocator.getPositionStream(locationSettings: _settings).listen(
      (pos) {
        // Geolocator speed is m/s; convert to km/h
        final kmh = (pos.speed.isFinite ? pos.speed : 0.0) * 3.6;
        _push(kmh);
      },
      onError: (_) {
        // Ignore individual errors to keep stream alive
      },
    );
  }

  void _push(double kmh) {
    // Simple moving average smoothing
    _buf.add(kmh);
    if (_buf.length > _window) _buf.removeAt(0);

    final avg = _buf.reduce((a, b) => a + b) / _buf.length;
    latestKmh = avg;
    if (!_controller.isClosed) _controller.add(latestKmh);
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
    _buf.clear();
  }

  Future<void> dispose() async {
    await stop();
    await _controller.close();
  }
}
