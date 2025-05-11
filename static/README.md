## Step 1: Upload Images
- 📁 **Supported Types:** B-mode images and CEUS video (PNG / JPG / MP4)
- 🖼️ You can upload multiple images at once, drag & drop supported.
- 📏 Please make sure image resolution is **400×600** for best results.
- 🔍 Preview images before submission to avoid input mistakes.
- 🆔 File names must be anonymized (no patient info).
- ⏱️ CEUS video should be at least **10 seconds** long to capture perfusion.
- 📌 _Tip:_ The system auto-selects key CEUS frames for optimal performance.

---

## Step 2: Run AI Inference
- 🤖 Click **Run Diagnosis** to launch the model.
- 🧠 The model performs tumor segmentation and classifies lesion as **benign or malignant**.
- 🩺 Extracted features include:
    - 📈 Time-intensity curve
    - 📊 Peak intensity & time-to-peak
    - 🎯 Tumor center & size
- ⚡ Inference takes ~5s per case and runs on either local or cloud backend.
- 🔎 Results will be visualized immediately after processing completes.

---

## Step 3: Generate Diagnostic Report
- 📋 All results are compiled into a structured, printable report.
- 🖼️ Includes original image with overlay, tumor location, and AI classification.
- 📂 One-click export available in PDF and JSON formats.
- 🧾 Each report has a unique case ID and is automatically saved in the case archive.
- 🔐 Reports can be securely shared with medical teams or uploaded to patient records.
- 🛠️ _Tip:_ You can customize the report format in the **Settings** panel.
