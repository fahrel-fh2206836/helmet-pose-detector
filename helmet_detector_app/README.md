# RideSafe - Helmet Pose Detection App

A Flutter mobile application that uses AI-powered computer vision to detect when motorcycle riders are looking at their phones while riding. The app provides real-time alerts to promote safer riding practices.

## 🚀 Features

- **Real-time Helmet Pose Detection**: Uses TensorFlow Lite models to analyze camera feed and detect if rider is looking at phone
- **Speed-based Alert System**: Adjusts alert timing based on current riding speed
- **GPS Speed Tracking**: Monitors riding speed using GPS and accelerometer data
- **Smart Notifications**: Sends alerts only when appropriate (speed > 5 km/h, sustained detection)
- **Live Camera Preview**: Real-time camera feed with detection overlay
- **Permission Management**: Handles camera, location, and notification permissions
- **Background Processing**: Heavy AI processing runs on background threads to maintain smooth UI

## 📱 Mobile Interactions and Logics

The app features:

- Live camera preview showing detection status
- Real-time speed display
- Phone detection status indicator
- Service activation/deactivation toggle
- Technical debug information panel

## 🛠️ Technical Details

### Used AI Model

- **Model Type**: TensorFlow Lite (TFLite) model for pose classification
- **Input**: 300x300 RGB images
- **Classes**: "looking" vs "not_looking"
- **Preprocessing**: YUV420 → RGB conversion, resize to 320x320, center crop to 300x300, PyTorch normalization
- **Inference**: Runs on background isolate threads for performance

### Performance Optimizations

- **Background Processing**: Heavy preprocessing runs on separate isolates
- **Rate Limiting**: Configurable FPS limit (default: 8 FPS)
- **Memory Management**: Efficient YUV420 to RGB conversion with zero-copy transfers
- **NNAPI Acceleration**: Uses Android Neural Networks API when available

### Speed Detection

- **Multi-source**: GPS and accelerometer data fusion
- **Smoothing**: Exponential moving average with configurable parameters
- **Accuracy Filtering**: Ignores readings with poor GPS accuracy
- **Dynamic Thresholds**: Alert timing varies based on speed ranges

## 📋 Requirements

- **Platform**: Android (iOS support future releases)
- **Flutter**: SDK ^3.8.0
- **Permissions**: Camera, Location, Notifications
- **Hardware**: Front-facing camera, GPS capability, IMU sensors

## 🚀 Getting Started

### Prerequisites

- Flutter SDK installed
- Android development environment set up
- Physical Android device (camera required)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd helmet_detector_app
   ```

2. **Install dependencies**

   ```bash
   flutter pub get
   ```

3. **Add model files**

   - Ensure `mobilenetv2_100_full_int8.tflite` is in `assets/` folder
   - Optional: Add other model variants for testing

4. **Run the app**
   ```bash
   flutter run
   ```

### First Launch

1. Grant all requested permissions (Camera, Location, Notifications)
2. Position device to capture your face/helmet area
3. Start riding and activate detection service
4. App will monitor for phone usage and send alerts when appropriate

## 🔧 Configuration

### Model Selection

The app uses `mobilenetv2_100_full_int8.tflite` by default. To use a different model:

```dart
_model = await HelmetPose.load(
  assetPath: 'assets/your_model.tflite'
);
```

### Performance Tuning

- **Inference FPS**: Adjust `maxFps` parameter in `HelmetVideoClassifier`
- **Speed Smoothing**: Modify `windowSize` and `emaAlpha` in `SpeedService`
- **Alert Thresholds**: Customize `_probThreshold`, `_minSpeedForAlert`, and `_cooldown`

## 📁 Project Structure

```
lib/
├── main.dart                 # App entry point and initialization
├── models/
│   └── helmet_pose.dart     # TFLite model wrapper and inference
├── screens/
│   └── main_screen.dart     # Main UI and service coordination
├── services/
│   ├── helmet_video_classifier.dart  # Camera stream and AI processing
│   ├── speed_services.dart           # GPS/accelerometer speed tracking
│   ├── permission_service.dart       # Permission management
│   ├── noti_service.dart             # Local notifications
|   └── preprocessing_service.dart    # Preprocessing done in isolate
└── widgets/
    ├── icon_with_text.dart           # Reusable UI components
    └── live_camera_preview.dart     # Camera preview widget
```

## 🔍 How It Works

1. **Camera Stream**: Captures YUV420 frames from front camera
2. **Preprocessing**: Converts to RGB, resizes, crops, and normalizes on background thread
3. **AI Inference**: Runs TFLite model to classify pose as "looking" or "not_looking"
4. **Speed Monitoring**: Tracks GPS/accelerometer data for current speed
5. **Alert Logic**: Sends notifications when:
   - Pose classified as "looking" with confidence > 55%
   - Speed > 5 km/h
   - Detection sustained for speed-dependent duration
   - Not in cooldown period (10 seconds)

## 📄 License

This project is part of a research initiative. Please check with the project maintainers for licensing details.
