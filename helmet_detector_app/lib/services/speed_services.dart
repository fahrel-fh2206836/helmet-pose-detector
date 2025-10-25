import 'dart:async';
import 'dart:math' as math;

import 'package:geolocator/geolocator.dart';
import 'package:helmet_detector_app/enum/common_enums.dart';
import 'package:sensors_plus/sensors_plus.dart';



/// Smoothed speed service with:
///  - sliding window (moving average)
///  - exponential smoothing (EMA)
///  - outlier rejection
///  - fallback delta-distance speed
///  - conservative IMU stub (fills short gaps only)
class SpeedService {
  // ---- Public API ----
  final _speedCtrl = StreamController<double>.broadcast();
  final _sourceCtrl = StreamController<SpeedSource>.broadcast();
  final _statusCtrl = StreamController<SpeedStatus>.broadcast();

  Stream<double> get speedStream => _speedCtrl.stream;
  Stream<SpeedSource> get sourceStream => _sourceCtrl.stream;
  Stream<SpeedStatus> get statusStream => _statusCtrl.stream;

  double latestKmh = 0.0;
  SpeedSource latestSource = SpeedSource.unknown;
  SpeedStatus status = SpeedStatus.stopped;

  // ---- Config ----
  final LocationSettings locationSettings;
  final int windowSize; // sliding window length (e.g., 4–8)
  final double emaAlpha; // 0..1, higher = more reactive
  final double minValidKmh; // below this, treat as 0 km/h
  final double maxJumpKmh; // reject spikes above this jump vs last
  final double maxAccuracyMeters; // reject GPS points with worse accuracy
  final Duration imuGapMax; // max time to use IMU estimation

  SpeedService({
    LocationSettings? locationSettings,
    this.windowSize = 5,
    this.emaAlpha = 0.35,
    this.minValidKmh = 0.3, // ~0.08 m/s
    this.maxJumpKmh = 80.0, // reject absurd single-sample jumps
    this.maxAccuracyMeters = 25.0, // drop poor accuracy fixes
    this.imuGapMax = const Duration(seconds: 2),
  }) : locationSettings =
           locationSettings ??
           const LocationSettings(
             accuracy: LocationAccuracy.bestForNavigation,
             distanceFilter: 2, // meters
             timeLimit: null,
           );

  // ---- Internals ----
  StreamSubscription<Position>? _gpsSub;
  StreamSubscription<AccelerometerEvent>? _accelSub;
  final List<double> _window = <double>[];
  Position? _prevPos;
  DateTime? _lastFixAt;
  DateTime? _startedAt;

  // IMU integration (very conservative)
  // We keep a tiny running integral of forward acceleration magnitude.
  // This is just a stub to keep UI responsive during <2s GPS gaps.
  double _imuKmhEstimate = 0.0;
  DateTime? _imuStart;

  // ---- Lifecycle ----
  Future<void> start() async {
    // Permissions & services
    final enabled = await Geolocator.isLocationServiceEnabled();
    if (!enabled) {
      _setStatus(SpeedStatus.gpsUnavailable);
      // still start IMU so UI can show something if desired
    }
    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied ||
        perm == LocationPermission.deniedForever) {
      _setStatus(SpeedStatus.permissionDenied);
      return;
    }

    _startedAt = DateTime.now();
    _setStatus(SpeedStatus.running);

    // GPS stream
    _gpsSub ??= Geolocator.getPositionStream(locationSettings: locationSettings)
        .listen(
          _onPosition,
          onError:
              (_) {}, // keep silent; app can observe statusStream for issues
        );

    // IMU stream (accelerometer)
    _accelSub ??= accelerometerEventStream().listen(_onAccel, onError: (_) {});

    // Warm start
    _emit(0.0, SpeedSource.unknown);
  }

  Future<void> stop() async {
    await _gpsSub?.cancel();
    await _accelSub?.cancel();
    _gpsSub = null;
    _accelSub = null;
    _prevPos = null;
    _window.clear();
    _imuKmhEstimate = 0.0;
    _imuStart = null;
    _setStatus(SpeedStatus.stopped);
  }

  Future<void> dispose() async {
    await stop();
    await _speedCtrl.close();
    await _sourceCtrl.close();
    await _statusCtrl.close();
  }

  // ---- Handlers ----
  void _onPosition(Position pos) {
    _lastFixAt = DateTime.now();

    // Reject poor accuracy fixes
    final horizontalAcc = pos.accuracy; // meters (if provided by platform)
    if (horizontalAcc.isFinite && horizontalAcc > maxAccuracyMeters) {
      return;
    }

    // Primary: GPS velocity (m/s)
    double? kmh;
    SpeedSource src = SpeedSource.gpsVelocity;

    if (pos.speed.isFinite && pos.speed >= 0) {
      kmh = pos.speed * 3.6;
    }

    // Fallback: distance / dt
    if (kmh == null || kmh.isNaN) {
      final prev = _prevPos;
      if (prev != null && pos.timestamp != null && prev.timestamp != null) {
        final dt =
            pos.timestamp!.difference(prev.timestamp!).inMilliseconds / 1000.0;
        if (dt > 0.2) {
          final d = Geolocator.distanceBetween(
            prev.latitude,
            prev.longitude,
            pos.latitude,
            pos.longitude,
          ); // meters
          kmh = (d / dt) * 3.6;
          src = SpeedSource.gpsDelta;
        }
      }
    }

    // If still null, keep last (do not emit). IMU may fill tiny gaps.
    if (kmh == null || kmh.isNaN) {
      _prevPos = pos;
      return;
    }

    // Clamp small noise to zero
    if (kmh.abs() < minValidKmh) kmh = 0.0;

    // Reject absurd spikes vs last
    if ((kmh - latestKmh).abs() > maxJumpKmh && latestKmh > 0) {
      _prevPos = pos;
      return;
    }

    // Reset IMU estimate on fresh GPS fix
    _imuKmhEstimate = kmh;
    _imuStart = DateTime.now();

    // Smooth and emit
    final smoothed = _smooth(kmh);
    _emit(smoothed, src);

    _prevPos = pos;
  }

  void _onAccel(AccelerometerEvent e) {
    // If no recent GPS (< imuGapMax), conservatively propagate IMU speed
    final now = DateTime.now();
    if (_lastFixAt == null) return;
    final gap = now.difference(_lastFixAt!);
    if (gap > imuGapMax) return; // we don't trust longer IMU runs => drift

    // Very rough magnitude-only integration (not frame-aligned, but ok for tiny gaps).
    // NOTE: Without device orientation & gravity removal this is crude.
    // Keep this conservative to avoid drift explosions.
    final dt = 1 / 50.0; // assume ~50 Hz average; sensors_plus varies by device
    final g = 9.80665;
    final ax = e.x, ay = e.y, az = e.z;
    final aMag =
        math.sqrt(ax * ax + ay * ay + az * az) - g; // naive gravity removal

    // Integrate acceleration to velocity (m/s), then to km/h. Clamp to >=0.
    final dvMs = aMag * dt;
    final vMs = math.max(0.0, (_imuKmhEstimate / 3.6) + dvMs);
    _imuKmhEstimate = vMs * 3.6;

    // Blend IMU with latestKmh using small alpha so it only nudges UI
    final blended = (0.85 * latestKmh) + (0.15 * _imuKmhEstimate);
    final smoothed = _smooth(blended);

    // Don’t emit if we actually have a fresh GPS fix in this same instant
    if (_lastFixAt != null &&
        now.difference(_lastFixAt!) <= const Duration(milliseconds: 100)) {
      return;
    }

    _emit(smoothed, SpeedSource.imuEstimate);
  }

  // ---- Smoothing ----
  double _smooth(double kmh) {
    // Sliding window average
    _window.add(kmh);
    if (_window.length > windowSize) _window.removeAt(0);
    final avg = _window.reduce((a, b) => a + b) / _window.length;

    // EMA on top
    final ema = (emaAlpha * avg) + ((1 - emaAlpha) * latestKmh);
    return ema;
  }

  // ---- Emit helpers ----
  void _emit(double kmh, SpeedSource src) {
    latestKmh = kmh;
    latestSource = src;
    if (!_speedCtrl.isClosed) _speedCtrl.add(latestKmh);
    if (!_sourceCtrl.isClosed) _sourceCtrl.add(latestSource);
  }

  void _setStatus(SpeedStatus s) {
    status = s;
    if (!_statusCtrl.isClosed) _statusCtrl.add(status);
  }
}
