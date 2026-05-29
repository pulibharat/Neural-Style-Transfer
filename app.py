import os
import threading
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import InputRequired
from PIL import Image
from torchvision import transforms
import io

# Import your existing AdaIN code
from utils.model import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
VGG_WEIGHT_PATH = os.path.join(WEIGHTS_DIR, 'vgg_normalised.pth')
DECODER_WEIGHT_PATH = os.path.join(WEIGHTS_DIR, 'decoder_2.pth')

encoder = None
decoder = None
_models_lock = threading.Lock()


def ensure_weights():
    """Fetch weights at startup if missing (e.g. on Render where weights/ is not in git)."""
    if os.path.isfile(VGG_WEIGHT_PATH) and os.path.isfile(DECODER_WEIGHT_PATH):
        return
    try:
        from scripts.download_weights import main as download_weights
        if download_weights() != 0:
            print("Weight download failed.")
    except Exception as exc:
        print(f"Could not download weights: {exc}")


def load_decoder_checkpoint(decoder, path):
    """Load decoder weights from train.py (net.* keys) or net-only checkpoints (1.weight, ...)."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    if not any(k.startswith('net.') for k in state):
        try:
            decoder.net.load_state_dict(state)
            return
        except RuntimeError:
            state = {f'net.{k}': v for k, v in state.items()}

    decoder.load_state_dict(state)


def load_models():
    if not os.path.exists(VGG_WEIGHT_PATH) or not os.path.exists(DECODER_WEIGHT_PATH):
        raise FileNotFoundError(
            'Missing model weights. Please place vgg_normalised.pth and decoder_2.pth in the weights/ folder.'
        )

    enc = VGGEncoder(VGG_WEIGHT_PATH).to(device)
    dec = Decoder().to(device)
    load_decoder_checkpoint(dec, DECODER_WEIGHT_PATH)
    enc.eval()
    dec.eval()
    return enc, dec


def get_models():
    """Load models on first use so gunicorn can bind PORT before heavy download/load."""
    global encoder, decoder
    if encoder is not None and decoder is not None:
        return encoder, decoder
    with _models_lock:
        if encoder is not None and decoder is not None:
            return encoder, decoder
        ensure_weights()
        encoder, decoder = load_models()
        return encoder, decoder


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(
            content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        if form.validate_on_submit():
            if form.content.data and form.content.data.filename:
                if allowed_file(form.content.data.filename):
                    content_filename = secure_filename(
                        form.content.data.filename)
                    form.content.data.save(os.path.join(
                        app.config['UPLOAD_FOLDER'], content_filename))
                    form.content_path.data = content_filename
            else:
                content_filename = form.content_path.data

            if form.style.data and form.style.data.filename:
                if allowed_file(form.style.data.filename):
                    style_filename = secure_filename(form.style.data.filename)
                    form.style.data.save(os.path.join(
                        app.config['UPLOAD_FOLDER'], style_filename))
                    form.style_path.data = style_filename
            else:
                style_filename = form.style_path.data

            if content_filename and style_filename:
                try:
                    enc, dec = get_models()
                    content_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], content_filename)
                    style_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], style_filename)

                    content_image = Image.open(content_path).convert('RGB')
                    style_image = Image.open(style_path).convert('RGB')

                    alpha = float(form.alpha.data)
                    stylized_image = style_transfer(
                        content_image, style_image, enc, dec, alpha, device)

                    result_filename = 'stylized_' + content_filename
                    result_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], result_filename)
                    save_image(stylized_image, result_path)
                    result_image = result_filename
                except (FileNotFoundError, RuntimeError) as e:
                    error = (
                        'Model weights failed to load on the server. '
                        'Redeploy the latest code from GitHub (main), then check Render logs. '
                        f'Details: {e}'
                    )
                except Exception as e:
                    error = str(e)
            else:
                error = 'Please upload both a content image and a style image.'

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/health')
def health():
    return 'ok', 200


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('Demo_IO_Images', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('localhost', 5000, app, use_reloader=True, use_debugger=True)
