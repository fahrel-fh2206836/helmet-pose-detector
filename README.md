# Helmet Pose Detector (Looking vs. Not Looking)

A lightweight computer-vision project that detects whether a helmeted rider is **looking at a phone** (*looking*) or **not** (*not looking*) to help identify and reduce mobile distractions.  
The classifier is a **CNN** fine-tuned via **transfer learning (EfficientNet-B3)** and runs in real time with a simple **OpenCV** UI.

## Coming Soon!

📱 **Mobile App in development**: A companion mobile application is being built to extend functionality to smartphones for on-road use.  

## Features
- **Binary attention classification:** *looking* vs. *not looking*  
- **Real-time feedback:** OpenCV window overlay for live camera/video  
- **Transfer learning:** EfficientNet-B3 fine-tuned on **1,800+ images** (roughly balanced)  
- **Portable stack:** Python + PyTorch + OpenCV (CPU works; CUDA optional)  
- **Mobile app in development:** A companion mobile application is being built to extend functionality to smartphones for on-road use. (Coming soon!)

---

## Quick Start

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
> Optional: if you have an NVIDIA GPU and want CUDA acceleration, install the CUDA-enabled wheels from PyTorch’s official instructions.

### 3) (Optional) VS Code interpreter selection
1. Open the project folder in VS Code (File → Open Folder… or `code .`)  
2. Press **Ctrl+Shift+P** → **Python: Select Interpreter**  
3. Choose the interpreter inside your venv, e.g.:
   - **Windows:** `venv\Scripts\python.exe`
   - **Unix/macOS:** `./venv/bin/python`

---

## Run the Demo

```bash
# From the repo root, with venv activated:
python helmet-model-test.py
```

**Notes:**
- If your script expects a **model weights path**, make sure the file exists and the path in the code matches (e.g., `models/efficientnet_b3.pth`).  
- If using a webcam, ensure the index (e.g., `cv2.VideoCapture(0)`) is correct.  
- Some environments need `python3` instead of `python`.

---

## Project Goal & Approach

- **Goal:** Detect rider distraction by classifying whether a helmeted user is looking at a phone.  
- **Model:** EfficientNet-B3 (pre-trained) fine-tuned with **PyTorch**.  
- **Data:** ~**1,800** labeled images, roughly half *looking* and half *not looking*.  
- **UI:** Simple **OpenCV** interface for real-time visualization.  
- **Mobile app:** Development is ongoing to deploy this model on smartphones for broader accessibility.  

---

## Repository Structure (typical)
```
helmet-pose-detector/
├─ helmet-test.py          # Run-time demo (camera/video + UI)
├─ train/                  # (If present) training scripts / notebooks
├─ models/                 # Saved weights (e.g., efficientnet_b3.pth)
├─ data/                   # (Optional) sample images or dataset pointers
├─ requirements.txt        # (Optional) pinned deps
└─ README.md
```

---

## Training (Overview)

> If you plan to re-train or fine-tune:
- Use **Google Colab** or local GPU.  
- Start from **EfficientNet-B3** pre-trained weights.  
- Split data into train/val (e.g., 80/20), maintain label balance.  
- Apply light augmentations (random crop/flip/brightness) to improve robustness.  
- Track metrics (accuracy, F1) and save the **best validation checkpoint**.

---

## Troubleshooting

- **OpenCV window doesn’t appear:** Ensure a valid camera index and that another app isn’t using the camera.  
- **CUDA errors:** Install a CUDA-compatible PyTorch build or run on CPU.  
- **Module not found / wrong interpreter:** Verify your **venv is activated** and VS Code is pointing to the correct interpreter.  
- **Weights not found:** Confirm the path/filename in code matches your actual weights file.

---

## Tech Stack
**Python, PyTorch, OpenCV, Pillow**  
(Optional: CUDA/cuDNN for GPU acceleration)

---

## License
This project is under MIT license. [License](https://github.com/fahrel-fh2206836/helmet-pose-detector/blob/main/LICENSE).
