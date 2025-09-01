# ⛑️ Helmet Pose Detector (Looking vs. Not Looking)

A lightweight computer-vision project that detects whether a helmeted rider is **looking at a phone** (*looking*) or **not** (*not looking*) to help identify and alert for the purpose of reducing mobile distractions.  
The classifier is a **CNN** fine-tuned via **transfer learning (EfficientNet-B3)** and runs in real time with a simple **OpenCV** UI.

## 🔜 Coming Soon!

📱 **Mobile App in development**: A companion mobile application is being built to extend functionality to smartphones for on-road use.  

## ✨ Features
- **Binary attention classification:** *looking* vs. *not looking*  
- **Real-time feedback:** OpenCV window overlay for live camera/video  
- **Transfer learning:** EfficientNet-B3 fine-tuned on **1,800+ images** (roughly balanced)  
- **Mobile app in development:** A companion mobile application is being built to extend functionality to smartphones for on-road use. (Coming soon!) 🛠️

---

## 🚩 Quick Start

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

---

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

## 🎯 Project Goal & Approach

- **Goal:** Detect rider distraction by classifying whether a helmeted user is looking at a phone.  
- **Model:** EfficientNet-B3 (pre-trained) fine-tuned with **PyTorch**.  
- **Data:** ~**1,800** labeled images, roughly half *looking* and half *not looking*.  
- **UI:** Simple **OpenCV** interface for real-time visualization.  
- **Mobile app:** Development is ongoing to deploy this model on smartphones for broader accessibility. **(Coming Soon!)** 🛠️
  
---

## License
This project is under MIT license. [License](https://github.com/fahrel-fh2206836/helmet-pose-detector/blob/main/LICENSE).
