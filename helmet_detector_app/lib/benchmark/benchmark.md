# 🧠 Device Specifications

| **Parameter**                 | **Samsung Galaxy S24+** | **Samsung Galaxy A52**   |
| :---------------------------- | :---------------------- | :----------------------- |
| **SoC / Chipset**             | Samsung Exynos 2400     | Qualcomm Snapdragon 720G |
| **CPU Architecture**          | ARMv8                   | ARMv8                    |
| **Cores**                     | 10 Cores                | 8 Cores                  |
| **Clock Speeds (GHz)**        | 0.4 - 3.2               | 0.3 - 2.3                |
| **RAM (Size)**                | 12 GB                   | 8 GB                     |
| **Android Version**           | 16                      | 14                       |
| **Battery Saver During Test** | Off                     | Off                      |

<br><br>

# ⚙️ Model & Benchmark Parameters

| **Parameter**      | **Value**                                   |
| :----------------- | :------------------------------------------ |
| Model A            | `helmet_pose_fp32.tflite` (41.75 MB)        |
| Model B            | `helmet_pose_fp32io_fp16.tflite` (20.97 MB) |
| Input Size         | 300×300 (cropped from 320×320)              |
| Runs               | 30                                          |
| Warmup             | 5                                           |
| Threads Tested     | 1, 2, 3, 4, 6, 8                            |
| Delegates Tested   | XNNPACK, NNAPI, Basic CPU                   |
| Metric Definitions | Avg, Min, Median, P90, Max, FPS             |

<br><br>

# 📊 Performance Results

> Note: NNAPI manges threading automatically.

| **Device**  | **Delegate**  | **Threads** | **Model (FP32)** <br> Avg / P90 / Max / FPS | **Model (FP32 I/O + FP16 Internal)** <br> Avg / P90 / Max / FPS |
| :---------- | :------------ | :---------: | :------------------------------------------ | :-------------------------------------------------------------- |
| **Phone A** | **XNNPACK**   |      1      |                                             |                                                                 |
|             |               |      2      |                                             |                                                                 |
|             |               |      3      |                                             |                                                                 |
|             |               |      4      |                                             |                                                                 |
|             |               |      6      |                                             |                                                                 |
|             |               |      8      |                                             |                                                                 |
|             | **Basic CPU** |      1      |                                             |                                                                 |
|             |               |      2      |                                             |                                                                 |
|             |               |      3      |                                             |                                                                 |
|             |               |      4      |                                             |                                                                 |
|             |               |      6      |                                             |                                                                 |
|             |               |      8      |                                             |                                                                 |
|             | **NNAPI**     |      –      |                                             |                                                                 |
| **Phone B** | **XNNPACK**   |      1      |                                             |                                                                 |
|             |               |      2      |                                             |                                                                 |
|             |               |      3      |                                             |                                                                 |
|             |               |      4      |                                             |                                                                 |
|             |               |      6      |                                             |                                                                 |
|             |               |      8      |                                             |                                                                 |
|             | **Basic CPU** |      1      |                                             |                                                                 |
|             |               |      2      |                                             |                                                                 |
|             |               |      3      |                                             |                                                                 |
|             |               |      4      |                                             |                                                                 |
|             |               |      6      |                                             |                                                                 |
|             |               |      8      |                                             |                                                                 |
|             | **NNAPI**     |      –      |                                             |                                                                 |
