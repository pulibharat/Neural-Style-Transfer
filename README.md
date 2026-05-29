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
- Model weights are not in git. On deploy, weights are downloaded automatically from the [AdaIN release](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0) on first startup (or run `python scripts/download_weights.py` during build).
- Optional Render build command: `pip install -r requirements.txt && python scripts/download_weights.py`
- If deploying to Heroku or Render, the included `Procfile` uses `gunicorn`.
- Render defaults to Python 3.14; this app needs Python 3.12 for `torch==2.2.2`.
- Pin Python with a `.python-version` file containing `3.12` (do not use `3.12.17`; Render cannot install it).
- On Render, use **Manual Deploy → Deploy latest commit** so the build picks up the newest `main`, not an old failed deploy.

## Optional training

If you want to train new weights, run:

```powershell
python train.py --batch_size 4 --epochs 10 --experiment big_Data --content_dir content_img --style_dir style_img
```

If you use a custom weights location, update `app.py` or pass `--vgg`.
