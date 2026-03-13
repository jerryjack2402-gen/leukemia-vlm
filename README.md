# Leukemia VLM: Vision-Language Model for Leukemia Detection

A comprehensive Vision-Language Large Language Model (VLM-LLM) framework designed for automated leukemia detection and pathology reporting from blood smear images.

## Features

- **Vision Encoder**: Extracts robust visual features from high-resolution blood smear images.
- **Language Model Generator**: Generates detailed, clinical-grade automated pathology reports.
- **Multimodal Fusion**: Combines visual features with text embeddings for contextual understanding.
- **Web Interface**: Easy-to-use application (`app.py`) for uploading images and generating reports.
- **Training Pipeline**: Complete training script (`train.py`) to fine-tune the model on custom datasets.

## Project Layout

```
leukemia_vlm/
├── app.py               # Web interface (e.g., Streamlit/Gradio)
├── train.py             # Model training script
├── inference.py         # Standalone script for running predictions
├── requirements.txt     # Project dependencies
├── run_app.bat          # Batch script to launch the app on Windows
├── models/              # Model architecture definitions
│   ├── vision_encoder.py
│   ├── llm_generator.py
│   └── fusion.py
├── data/                # Data loaders and dataset definitions
│   └── dataset.py
└── utils/               # Helper utilities
```

## Installation

1. **Clone the repository** (or download the source):
   ```bash
   git clone <your-github-repo-url>
   cd leukemia_vlm
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web App
You can start the web interface by running:
```bash
python app.py
```
*(Alternatively, on Windows, just double-click `run_app.bat`)*

### Training the Model
To start training the model on your dataset:
```bash
python train.py
```

### Running Inference
To generate a structured pathology report for a specific image:
```bash
python inference.py --image_path sample_blood_smear.jpg
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
