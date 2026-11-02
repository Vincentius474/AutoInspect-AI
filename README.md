# AutoInspect AI

<img width="1863" height="937" alt="Screenshot 2026-11-02 181340" src="https://github.com/user-attachments/assets/deedb8bb-bcd3-4dab-abd9-1e3888bfd904" />

<img width="1581" height="840" alt="Screenshot 2026-11-02 181537" src="https://github.com/user-attachments/assets/85584f11-2fb8-4098-adef-cee0fe7f0ed5" />

<img width="1472" height="911" alt="Screenshot 2026-11-02 181604" src="https://github.com/user-attachments/assets/46cb3ce5-6d24-4454-9b31-2124e01f39fe" />

AutoInspect AI classifies visible vehicle damage into three severity levels:
**Low**, **Medium**, and **High**. It includes MobileNetV3-Small training,
evaluation scripts, and a Streamlit app with Grad-CAM explanations.

The prediction is an estimate of visible damage only. It does not assess
hidden structural damage, roadworthiness, or repair costs.

## Setup

Use Python 3.10 or newer. From the repository root, create an environment and
install the packages listed in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

Place the source images in these class folders:

```text
data/raw/
  training/
    01-minor/
    02-moderate/
    03-severe/
  validation/
    01-minor/
    02-moderate/
    03-severe/
```

Create the model-ready splits with:

```powershell
python -m src.data.splitting
```

The script reserves a stratified portion of the training images for testing
and preserves the supplied validation split. It refuses to overwrite a
non-empty `data/splits` directory.

## Train and evaluate

MobileNetV3-Small training settings are in `configs/config.yaml`. Start
training with:

```powershell
python -m src.training.train_mobilenet
```

The best checkpoint is written to
`models/checkpoints/best_mobilenet.pth`. Evaluate it on the held-out test
split with:

```powershell
python -m src.evaluation.run_test_evaluation
```

Dataset preparation and evaluation scripts expect to be run from the
repository root.

## Run the application

The Streamlit app uses the MobileNetV3-Small checkpoint:

```powershell
streamlit run app/streamlit_app.py
```

Upload a JPG, JPEG, or PNG image to see the severity estimate and class
probabilities. The Grad-CAM option can display the strongest regions of
influence as contour outlines. These outlines are not damage boundaries or
proof that a prediction is correct.

## Deploy to Streamlit Community Cloud

This project is a Streamlit app, so deploy it to Streamlit Community Cloud
rather than as a Vercel Python function:

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app using the repository and branch.
3. Set the main file path to `app/streamlit_app.py`.
4. Deploy with Python 3.10 or newer.

The standalone Grad-CAM script can also save a visualization for a local
image:

```powershell
python -m src.evaluation.gradcam path\to\vehicle.jpg
```

## Project layout

```text
app/                  Streamlit interface
configs/              Training configuration
data/                 Source images and generated dataset splits
models/checkpoints/   Trained model checkpoints
notebooks/            Dataset exploration
reports/              Evaluation outputs and experiment notes
src/data/             Dataset loading, splitting, and preprocessing
src/evaluation/       Evaluation and Grad-CAM utilities
src/models/           Model definitions
src/training/         Training loops and metrics
tests/                Automated regression tests
```

Run the automated tests with Python's built-in unittest runner:

```powershell
python -m unittest discover -s tests -v
```
