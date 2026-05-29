"""Download AdaIN model weights into weights/ (for local dev and Render builds)."""

import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Official AdaIN release assets (naoto0804/pytorch-AdaIN)
VGG_URL = os.environ.get(
    "VGG_WEIGHT_URL",
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth",
)
DECODER_URL = os.environ.get(
    "DECODER_WEIGHT_URL",
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth",
)

FILES = (
    ("vgg_normalised.pth", VGG_URL),
    # App expects decoder_2.pth; use pretrained decoder.pth if you have no custom checkpoint.
    ("decoder_2.pth", DECODER_URL),
)


def download_file(url: str, dest: str) -> None:
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  -> saved {dest} ({size_mb:.1f} MB)")


def main() -> int:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for filename, url in FILES:
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            print(f"Already present: {dest}")
            continue
        try:
            download_file(url, dest)
        except Exception as exc:
            print(f"Failed to download {filename}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
