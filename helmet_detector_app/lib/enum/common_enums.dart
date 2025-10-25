/// How the current value was produced (for debugging/telemetry).
enum SpeedSource { gpsVelocity, gpsDelta, imuEstimate, unknown }

/// Status of the tracker.
enum SpeedStatus { stopped, running, gpsUnavailable, permissionDenied }

String prettySource(SpeedSource s) {
  switch (s) {
    case SpeedSource.gpsVelocity:
      return 'GPS velocity';
    case SpeedSource.gpsDelta:
      return 'GPS distance/dt';
    case SpeedSource.imuEstimate:
      return 'IMU estimate';
    case SpeedSource.unknown:
      return '—';
  }
}

String prettyStatus(SpeedStatus s) {
  switch (s) {
    case SpeedStatus.running:
      return 'Running';
    case SpeedStatus.gpsUnavailable:
      return 'GPS unavailable';
    case SpeedStatus.permissionDenied:
      return 'Permission denied';
    case SpeedStatus.stopped:
      return 'Stopped';
  }
}
