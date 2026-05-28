# Neural Style Transfer Flask App

This repository contains a Flask-based demo for neural style transfer and the training code used to create the model.

## What to push to GitHub

Push the deployable code and UI files only:
- `app.py`
- `train.py`
- `utils/`
- `templates/`
- `static/`
- `Demo_IO_Images/`
- `requirements.txt`
- `.gitignore`
- `README.md`
- `Procfile` (optional for Heroku)

## What not to push

Do not push large or environment-specific files:
- `venv/`
- `experiment/`
- `content_data/`
- `content_img/`
- `style_data/`
- `style_img/`
- `mana/` (nested saved-model folder)
- `weights/`
- `*.pth` / `*.pt`

These files are excluded by `.gitignore`.

## Repository structure

```
/ (root)
  app.py
  train.py
  requirements.txt
  README.md
  Procfile
  .gitignore
  templates/
  static/
  utils/
  Demo_IO_Images/
  weights/  # not tracked by git
```

## Local setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Create the `weights/` directory and add required model files:

```powershell
mkdir weights
```

Required files:
- `weights/vgg_normalised.pth`
- `weights/decoder_2.pth`

These weight files are intentionally not tracked in Git to keep the repository small.

## Run the Flask app

```powershell
python app.py
```

Then open:

```
http://localhost:5000
```

## Deployment notes

- Push only the code, UI, and config files listed above.
- Add model weights to the deployment environment separately.
- If deploying to Heroku or Render, the included `Procfile` uses `gunicorn`.
- Render defaults to Python 3.14, but this app needs Python 3.12 for `torch==2.5.1`.
- Add a `runtime.txt` file with `python-3.12.17` to pin the Python version on Render.

## Optional training

If you want to train new weights, run:

```powershell
python train.py --batch_size 4 --epochs 10 --experiment big_Data --content_dir content_img --style_dir style_img
```

If you use a custom weights location, update `app.py` or pass `--vgg`.
