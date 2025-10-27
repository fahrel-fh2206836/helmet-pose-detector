# ⛑️ Helmet Pose Detector (Looking vs. Not Looking)

A lightweight computer-vision project that detects whether a helmeted rider is **looking at a phone** (_looking_) or **not** (_not looking_) to help identify and alert for the purpose of reducing mobile distractions.  
The classifier is a **CNN** fine-tuned via **transfer learning (EfficientNet-B3)** and runs in real time with a simple **OpenCV** UI as well as **Mobile** app.

## 🔜 Coming Soon!
🔙 **Background Operation**: In future releases, the app’s detection logic will run seamlessly in the background, allowing continuous monitoring even while the user is using other apps — a more practical setup for real on-road driving scenarios.

## 🆕 NEW UPDATES!

📱 **Mobile App Development – Completed:** Version 1 of the companion mobile application has been fully developed, extending the system’s functionality to smartphones for seamless on-road use.

⚠️ **Note:** The mobile application is currently available only for **Android**. iOS support will be added in a future release.

## ✨ Features

- **Binary attention classification:** _looking_ vs. _not looking_
- **Real-time feedback:** OpenCV window overlay for live camera/video
- **Transfer learning:** EfficientNet-B3 fine-tuned on **1,800+ images** (roughly balanced)
- **Mobile app:** Version 1 of the companion mobile application has been fully developed, extending the system’s functionality to smartphones for seamless on-road use. 🆕

## Git Clone Repository

```
git clone https://github.com/fahrel-fh2206836/helmet-pose-detector.git
cd helmet-pose-detector
```

## 🚩 Quick Start for OpenCV

### 1) Create & activate a virtual environment

```bash
# Windows (PowerShell / CMD)
python -m venv venv
venv\Scripts\activate

# Unix / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install torch torchvision opencv-python pillow
```

### 3) VS Code interpreter selection

1. Open the project folder in VS Code (File → Open Folder… or `code .`)
2. Press **Ctrl+Shift+P** → **Python: Select Interpreter**
3. Choose the interpreter inside your venv, e.g.:
   - **Windows:** `venv\Scripts\python.exe`
   - **Unix/macOS:** `./venv/bin/python`

## 🚀 Run the Demo

```bash
# Open the folder with the myenv (The environment you just created) folder created, with venv activated:
python helmet-pose-detector/open_cv_ui/helmet-model-test.py # Basically the path to helmet-model-test.py from the folder you opened in VS Code
```

**Notes:**

- If your script expects a **model weights path**, make sure the file exists and the path in the code matches (e.g., `models/efficientnet_b3.pth`).
- If using a webcam, ensure the index (e.g., `cv2.VideoCapture(0)`) is correct.
- Some environments need `python3` instead of `python`.

---

## 🚩 Quick Start and Setup for Mobile App

### 1) 📋 Check Prerequisites

Before you begin, make sure you have the following installed:

- Flutter SDK (version ≥ 3.0.0)
- Dart SDK (comes with Flutter)
- Android Studio or VS Code with Flutter extensions
- Android SDK & Emulator (or a physical Android device)

### 2) 📂 Open folder on IDE

```
cd helmet_detector_app
```

Ensure that you open the `helmet_detector_app` folder on your preferred IDE to prevent issues.

### 3) ⚙️ Install Dependencies

```
flutter pub get
```

### 4) 🚀 Run the App

```
flutter run
```

Make sure a device is connected and recognized by `flutter devices`.

**Note:** More Technical details can be found under `helmet_detector_app/readme.md`

## 🎯 Project Goal & Approach

- **Goal:** Detect rider distraction by classifying whether a helmeted user is looking at a phone.
- **Model:** EfficientNet-B3 (pre-trained) fine-tuned with **PyTorch**.
- **Data:** ~**1,800** labeled images, roughly half _looking_ and half _not looking_.
- **UI:** Simple **OpenCV** interface for real-time visualization.
- **Mobile app:** Development successfully completed, the model has been fully deployed on smartphones, ensuring broader accessibility and real-world usability. 🆕

## 🚨 Safety Notice

This app is designed to promote safer riding practices but should not be relied upon as the sole safety measure. Always:

- Keep your eyes on the road
- Use hands-free communication devices
- Follow local traffic laws
- Ride defensively and responsibly

## License

This project is under MIT license. [License](https://github.com/fahrel-fh2206836/helmet-pose-detector/blob/main/LICENSE).

## 🙏 Acknowledgements & Datasets

This project’s training and fine-tuning process utilized the following publicly available datasets:  
- [Bykea Helmet Dataset (Roboflow)](https://universe.roboflow.com/samsun-vwq9u/bykea)  
- [Helmet Detection Dataset (Roboflow)](https://universe.roboflow.com/myspace-60grk/helmet-jmvny)  
- [Motorcycle Helmet Dataset (Mendeley Data)](https://data.mendeley.com/datasets/bmy35m25pw/1)  
- [Safety Helmet Detection Dataset (Mendeley Data)](https://data.mendeley.com/datasets/tm72fkfxd5/3)  

> All datasets are credited to their respective authors and are used solely for research and educational purposes.

