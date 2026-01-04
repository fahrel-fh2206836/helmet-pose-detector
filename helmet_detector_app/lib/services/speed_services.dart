/* 
Smoothed speed tracking service with:
  - sliding window (moving average)
  - exponential smoothing (EMA)
  - outlier rejection
  - fallback delta-distance speed
  - conservative IMU stub (fills short gaps only)
*/

import 'dart:async';
import 'dart:math' as math;
import 'package:geolocator/geolocator.dart';
import 'package:helmet_detector_app/enum/common_enums.dart';
import 'package:sensors_plus/sensors_plus.dart';

class SpeedService {
  // Broadcasts values to listeners in the UI
  final _speedCtrl = StreamController<double>.broadcast();
  final _sourceCtrl = StreamController<SpeedSource>.broadcast();
  final _statusCtrl = StreamController<SpeedStatus>.broadcast();

  Stream<double> get speedStream => _speedCtrl.stream;
  Stream<SpeedSource> get sourceStream => _sourceCtrl.stream;
  Stream<SpeedStatus> get statusStream => _statusCtrl.stream;

  // Latest state for quick reads in the UI
  double latestKmh = 0.0;
  SpeedSource latestSource = SpeedSource.unknown;
  SpeedStatus status = SpeedStatus.stopped;

  // Configurations
  final LocationSettings locationSettings; // GPS subscription settings
  final int windowSize; // sliding window length (e.g., 4–8)
  final double emaAlpha; // 0..1, higher = more reactive
  final double minValidKmh; // below this, treat as 0 km/h
  final double maxAccuracyMeters; // reject GPS points with worse accuracy
  final Duration imuGapMax; // max time to use IMU estimation
  final Duration imuGapMin; // min time to use IMU estimation

  SpeedService({
    LocationSettings? locationSettings,
    this.windowSize = 5,
    this.emaAlpha = 0.6,
    this.minValidKmh = 0.3,
    this.maxAccuracyMeters = 25.0, // drop poor accuracy fixes
    this.imuGapMax = const Duration(seconds: 5),
    this.imuGapMin = const Duration(seconds: 2),
  }) : locationSettings =
           locationSettings ??
           const LocationSettings(
             accuracy: LocationAccuracy.bestForNavigation,
             distanceFilter: 2, // meters before next update
             timeLimit: null,
           );

  // Internals subscriptions
  StreamSubscription<Position>? _gpsSub;
  StreamSubscription<AccelerometerEvent>? _accelSub;
  final List<double> _window = <double>[];
  Position? _prevPos;
  DateTime? _lastFixAt;
  DateTime? _startedAt;

  /* 
  IMU integration (very conservative)
  We keep a tiny running integral of forward acceleration magnitude.
  This is just a stub to keep UI responsive during <2s GPS gaps.
  */
  double _imuKmhEstimate = 0.0;
  DateTime? _imuStart;

  // Lifecycle (Start speed trakcing service)
  Future<void> start() async {
    // Check if GPS service is enabled
    final enabled = await Geolocator.isLocationServiceEnabled();
    if (!enabled) {
      _setStatus(SpeedStatus.gpsUnavailable);
      // We still proceed to start IMU so UI show something minimal
    }

    // Check permissions
    var perm = await Geolocator.checkPermission();
    if (perm == LocationPermission.denied ||
        perm == LocationPermission.deniedForever) {
      _setStatus(SpeedStatus.permissionDenied);
      return;
    }

    _startedAt = DateTime.now();
    _setStatus(SpeedStatus.running);

    // Subscribe to GPS stream
    _gpsSub ??= Geolocator.getPositionStream(locationSettings: locationSettings)
        .listen(
          _onPosition,
          onError:
              (_) {}, // keep silent; app can observe statusStream for issues
        );

    // Subscribe to accelerometer (IMU) stream
    _accelSub ??= accelerometerEventStream().listen(_onAccel, onError: (_) {});

    // Emit an initial value so UI can render immediately
    _emit(0.0, SpeedSource.unknown);
  }

  // Stop speed tracking service to release resources
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

  // Destroy speed tracking service controllers (Removes completely from widget tree)
  Future<void> dispose() async {
    await stop();
    await _speedCtrl.close();
    await _sourceCtrl.close();
    await _statusCtrl.close();
  }

  // Handlers
  void _onPosition(Position pos) {
    _lastFixAt = DateTime.now();

    // Reject poor accuracy fixes
    final horizontalAcc = pos.accuracy; // meters
    if (horizontalAcc.isFinite && horizontalAcc > maxAccuracyMeters) {
      return;
    }

    // Primary: use GPS-provided speed when available (m/s -> km/h)
    double? kmh;
    SpeedSource src = SpeedSource.gpsVelocity;

    if (pos.speed.isFinite && pos.speed >= 0) {
      kmh = pos.speed * 3.6;
    }

    // Fallback: compute speed from distance / time between fixes
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

    // If still no speed, stash pos and wait (IMU may fill tiny gaps)
    if (kmh == null || kmh.isNaN) {
      _prevPos = pos;
      return;
    }

    // Clamp small noise to zero
    if (kmh.abs() < minValidKmh) kmh = 0.0;

    // Reset IMU estimate on fresh GPS fix
    _imuKmhEstimate = kmh;
    _imuStart = DateTime.now();

    // Smooth (window avg + EMA) and emit
    final smoothed = _smooth(kmh);
    _emit(smoothed, src);

    _prevPos = pos;
  }

  void _onAccel(AccelerometerEvent e) {
    // Only try IMU fill if the last GPS fix is recent enough
    final now = DateTime.now();

    // We can’t do anything without at least one GPS fix
    if (_lastFixAt == null) return;

    final gap = now.difference(_lastFixAt!);

    // ---- GAP LOGIC -------------------------------------------------
    // Only use IMU when:
    //   imuGapMin <= gap <= imuGapMax
    //
    // - gap < imuGapMin  => GPS is still "fresh": keep last GPS speed.
    // - gap > imuGapMax  => IMU would drift too much: don't trust it.

    // start IMU fill after ~1 s of silence
    if (gap < imuGapMin) {
      // GPS is updating normally, don't overwrite it with IMU
      return;
    }

    if (gap > imuGapMax) {
      // Gap too long: stop trusting IMU until next GPS fix
      return;
    }
    // ----------------------------------------------------------------

    // Very rough magnitude-only integration (not frame-aligned).
    // NOTE: Without device orientation & gravity removal this is crude.
    // Keep this conservative to avoid drift explosions.
    const dt = 1.0 / 50;
    const g = 9.80665;

    final ax = e.x;
    final ay = e.y;
    final az = e.z;

    final aMag =
        math.sqrt(ax * ax + ay * ay + az * az) - g; // naive gravity removal

    // v = v + a * dt (m/s), then convert to km/h, clamp to >= 0.
    final dvMs = aMag * dt;
    final vMs = math.max(0.0, (_imuKmhEstimate / 3.6) + dvMs);
    _imuKmhEstimate = vMs * 3.6;

    // Blend IMU with latestKmh using small alpha so it only nudges UI
    final blended = (0.85 * latestKmh) + (0.15 * _imuKmhEstimate);
    final smoothed = _smooth(blended);

    _emit(smoothed, SpeedSource.imuEstimate);
  }

  // Smoothing (windowed avg + EMA)
  double _smooth(double kmh) {
    // Sliding window average
    _window.add(kmh);
    if (_window.length > windowSize) _window.removeAt(0);
    final avg = _window.reduce((a, b) => a + b) / _window.length;

    // Exponential moving average on top for extra smoothing
    final ema = (emaAlpha * avg) + ((1 - emaAlpha) * latestKmh);
    return ema;
    // return avg;
  }

  // Emit helpers
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
