// Notification Services

import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:helmet_detector_app/services/permission_service.dart';

class NotiService {
  final notificationsPlugin = FlutterLocalNotificationsPlugin();

  bool _isInitialized = false;
  bool get isInitialized => _isInitialized;

  // Initializes and configures notification services for Android
  Future<void> initNotification() async {
    if (_isInitialized) return;

    const initSettingsAndroid = AndroidInitializationSettings(
      '@mipmap/ic_launcher',
    );
    const initSettings = InitializationSettings(android: initSettingsAndroid);

    await notificationsPlugin.initialize(initSettings);

    _isInitialized = true;
  }

  // Returns notification configuration details for Android notifications.
  NotificationDetails notificationDetails() {
    return const NotificationDetails(
      android: AndroidNotificationDetails(
        'alert_channel',
        'RideSafe Alerts',
        importance: Importance.max,
        priority: Priority.high,
      ),
    );
  }

  /* 
  Displays a notification using the given [id], [title], and [body].
  checks for notification permission before showing the notification.
  If the user has not granted permission, the function returns early.
  */
  Future<void> showNotification({
    int id = 0,
    String? title,
    String? body,
  }) async {
    if (!await PermissionService.hasNotificationPermission()) {
      return;
    }
    return notificationsPlugin.show(id, title, body, notificationDetails());
  }
}
