"""Download model weights into weights/ (for Render / fresh clones).

- vgg_normalised.pth: downloaded from the official AdaIN release if missing.
- decoder_2.pth: only downloaded when DECODER_WEIGHT_URL is set (your trained checkpoint).
  We never substitute the public decoder.pth for your custom decoder_2.pth.
"""

import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

VGG_URL = os.environ.get(
    "VGG_WEIGHT_URL",
    "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth",
)
# Your trained checkpoint (GitHub Release v1.0). Override with DECODER_WEIGHT_URL if needed.
DEFAULT_DECODER_URL = (
    "https://github.com/pulibharat/Neural-Style-Transfer/releases/download/v1.0/decoder_2.pth"
)
DECODER_URL = os.environ.get("DECODER_WEIGHT_URL", DEFAULT_DECODER_URL)


def download_file(url: str, dest: str) -> None:
    print(f"Downloading {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  -> saved {dest} ({size_mb:.1f} MB)")


def main() -> int:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    failed = False

    vgg_dest = os.path.join(WEIGHTS_DIR, "vgg_normalised.pth")
    if os.path.isfile(vgg_dest) and os.path.getsize(vgg_dest) > 0:
        print(f"Already present: {vgg_dest}")
    else:
        try:
            download_file(VGG_URL, vgg_dest)
        except Exception as exc:
            print(f"Failed to download vgg_normalised.pth: {exc}", file=sys.stderr)
            failed = True

    decoder_dest = os.path.join(WEIGHTS_DIR, "decoder_2.pth")
    if os.path.isfile(decoder_dest) and os.path.getsize(decoder_dest) > 0:
        print(f"Already present: {decoder_dest}")
    elif DECODER_URL:
        try:
            download_file(DECODER_URL, decoder_dest)
        except Exception as exc:
            print(f"Failed to download decoder_2.pth: {exc}", file=sys.stderr)
            failed = True
    elif not DECODER_URL:
        print(
            "decoder_2.pth is missing. Set DECODER_WEIGHT_URL or add a GitHub Release asset.",
            file=sys.stderr,
        )
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
