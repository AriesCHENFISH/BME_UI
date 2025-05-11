Sure! Here's the complete translation of your project documentation into English, maintaining the original format:

---

# Accurate Diagnosis Method for Breast Tumors Based on Multimodal Ultrasound Dynamic Imaging

## Project Overview

This project aims to develop a precise breast tumor diagnosis system by combining multimodal ultrasound dynamic imaging (CEUS and B-mode) with deep learning methods. The system can provide accurate tumor classification and segmentation results through real-time dynamic ultrasound imaging, assisting in medical image analysis and improving diagnostic efficiency and accuracy.

## Project Structure

```
BME_UI/
├── ceus_dual_task/              # Ultrasound image processing and model code
│   ├── dual_tasks_code/         # Main task-related code
│   │   ├── __pycache__/         # Cache files
│   │   ├── segmentation/        # Segmentation model files
│   │   ├── segmentation_result/ # Segmentation results
│   │   ├── data.py              # Data loading and processing
│   │   ├── frozen_resnet50_lstm_model.py # Frozen ResNet50-LSTM model training
│   │   ├── main.py              # Main program
│   │   ├── model.py             # Model definition
│   │   ├── refer_ceus.py        # CEUS reference image processing
│   │   ├── refer.py             # Reference code
│   │   ├── test.py              # Testing code
│   │   ├── train.py             # Training code
│   ├── model_weights/           # Model weight files
│   ├── for_test/                # Test-related files
├── static/                      # Static files
│   ├── image/                   # Image resources
│   ├── info/                    # Information
│   ├── output/                  # Output files
│   ├── analytic.js              # Analysis script
│   ├── home.js                  # Homepage script
│   ├── README.md                # Project description file
│   ├── README.pdf               # Project description PDF
│   ├── script.js                # Script file
│   ├── start.css                # Page styles
│   ├── start.js                 # Page interaction script
│   ├── style.css                # Style file
│   ├── style2.css               # Auxiliary style
│   ├── style3.css               # Auxiliary style
├── templates/                   # HTML templates
│   ├── home.html                # Homepage template
│   ├── main.html                # Main interface template
│   ├── start.html               # Start page template
├── app.py                       # Flask application main program
```

## Functional Modules

1. **Image Classification and Segmentation Models**:

   * Uses ResNet50 as the encoder to extract spatial features from B-mode and CEUS images.
   * Uses UNet as the decoder for image segmentation tasks.
   * Applies LSTM to extract temporal features for image classification.
   * Dual-task joint training: simultaneously performs image classification and segmentation.

2. **Data Processing**:

   * The dataset contains breast tumor ultrasound images (including B-mode and CEUS) from Zhongda Hospital. Each sample includes 60 CEUS frames, one B-mode image, and a segmentation mask.
   * The dataset is split into training, validation, and testing sets in a 4:1:1 ratio.
   * Preprocessing includes resizing images to 224x224 and normalization.

3. **Training and Evaluation**:

   * Five-fold cross-validation is used to evaluate model performance.
   * Classification metrics include accuracy, precision, recall, F1 score, and confusion matrix.
   * Segmentation metrics include DICE coefficient, pixel accuracy, and Intersection over Union (IoU).

4. **Interface Design**:

   * Provides a user interface (UI) based on the Flask framework, which allows diagnostic result display, history record queries, and other functions.

## Installation and Usage

### Environment Requirements

* Python 3.6+
* Required Python packages: `torch`, `torchvision`, `flask`, `opencv-python`, `scikit-learn`, etc.

### Installation Steps

1. Clone the project:

   ```bash
   git clone <repository_url>
   cd BME_UI
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:

   ```bash
   python app.py
   ```

4. Access the project homepage:
   Open your browser and visit `http://127.0.0.1:5000/`.

## Project Progress

The following stages have been completed:

1. Collection and preprocessing of the dataset;
2. Preliminary model training and evaluation;
3. Design and implementation of the dual-task model based on B-mode and CEUS images.

## Future Plans

1. Further improve the B-mode branch network to enhance diagnostic accuracy;
2. Explore advanced fusion methods for CEUS and B-mode images, such as attention mechanisms;
3. Conduct in-depth research on spatiotemporal analysis-based model optimization to improve classification and segmentation precision;
4. Validate and deploy the model in real clinical environments.

---

Let me know if you’d like a formatted PDF or Markdown file of this translation.
