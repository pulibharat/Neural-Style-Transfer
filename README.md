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
- Model weights are not in git.
- **VGG** (`vgg_normalised.pth`): auto-downloaded from the [AdaIN release](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0) if missing.
- **Decoder** (`decoder_2.pth`): use **your trained checkpoint** — not the public `decoder.pth`.

### Deploy your trained `decoder_2.pth` on Render

1. On GitHub: **Releases → Create a new release** → attach `decoder_2.pth` from your local `weights/` folder.
2. Copy the release asset URL (right-click the file → copy link).
3. In Render → **Environment** → add `DECODER_WEIGHT_URL` = that URL.
4. Build command (recommended):

```bash
pip install -r requirements.txt && python scripts/download_weights.py
```

5. Redeploy with **Deploy latest commit**.
- If deploying to Heroku or Render, the included `Procfile` uses `gunicorn` with a 300s timeout.
- On Render, set **Start command** to match the Procfile (or leave blank to use Procfile):
  `gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 1 app:app`
- **Memory:** PyTorch + VGG needs more than **512 MB**. Free tier may hit `SIGKILL` / out of memory at 512px — use **Starter (1 GB+)** or set `INFERENCE_SIZE=384` (or `256`) in Environment.
- **Quality vs 502:** Render defaults to **384px** (local is **512px**). HTTP 502 on Generate usually means out-of-memory or timeout at 512px on free tier. For local-matching quality: upgrade to **1 GB+** and set `INFERENCE_SIZE=512`.
- Render defaults to Python 3.14; this app needs Python 3.12 for `torch==2.2.2`.
- Pin Python with a `.python-version` file containing `3.12` (do not use `3.12.17`; Render cannot install it).
- On Render, use **Manual Deploy → Deploy latest commit** so the build picks up the newest `main`, not an old failed deploy.

## Optional training

If you want to train new weights, run:

```powershell
python train.py --batch_size 4 --epochs 10 --experiment big_Data --content_dir content_img --style_dir style_img
```

If you use a custom weights location, update `app.py` or pass `--vgg`.
