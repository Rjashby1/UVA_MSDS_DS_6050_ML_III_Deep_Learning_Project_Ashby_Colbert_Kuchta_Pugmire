from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DATA"

DOWNLOADS = [
    # ISIC 2019
    ("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip",
     DATA / "raw" / "isic2019" / "ISIC_2019_Training_Input.zip"),
    ("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Metadata.csv",
     DATA / "isic2019" / "train" / "ISIC_2019_Training_Metadata.csv"),
    ("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv",
     DATA / "isic2019" / "train" / "ISIC_2019_Training_GroundTruth.csv"),
    ("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip",
     DATA / "raw" / "isic2019" / "ISIC_2019_Test_Input.zip"),
    ("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Metadata.csv",
     DATA / "isic2019" / "test" / "ISIC_2019_Test_Metadata.csv"),
    ("https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv",
     DATA / "isic2019" / "test" / "ISIC_2019_Test_GroundTruth.csv"),

    # MILK10k
    ("https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Input.zip",
     DATA / "raw" / "milk10k" / "MILK10k_Training_Input.zip"),
    ("https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Metadata.csv",
     DATA / "milk10k" / "train" / "MILK10k_Training_Metadata.csv"),
    ("https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Supplement.csv",
     DATA / "milk10k" / "train" / "MILK10k_Training_Supplement.csv"),
    ("https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_GroundTruth.csv",
     DATA / "milk10k" / "train" / "MILK10k_Training_GroundTruth.csv"),
    ("https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Test_Input.zip",
     DATA / "raw" / "milk10k" / "MILK10k_Test_Input.zip"),
    ("https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Test_Metadata.csv",
     DATA / "milk10k" / "test" / "MILK10k_Test_Metadata.csv"),
]

UNZIP_TARGETS = [
    (DATA / "raw" / "isic2019" / "ISIC_2019_Training_Input.zip", DATA / "isic2019" / "train" / "images"),
    (DATA / "raw" / "isic2019" / "ISIC_2019_Test_Input.zip",     DATA / "isic2019" / "test" / "images"),
    (DATA / "raw" / "milk10k"  / "MILK10k_Training_Input.zip",   DATA / "milk10k"  / "train" / "images"),
    (DATA / "raw" / "milk10k"  / "MILK10k_Test_Input.zip",       DATA / "milk10k"  / "test"  / "images"),
]

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Skip (exists): {dest}")
        return
    print(f"Downloading -> {dest.name}")
    urllib.request.urlretrieve(url, dest)

def unzip(zip_path: Path, out_dir: Path) -> None:
    if not zip_path.exists():
        print(f"Missing zip: {zip_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # quick “already unzipped?” check
    if any(out_dir.iterdir()):
        print(f"Skip unzip (non-empty): {out_dir}")
        return

    print(f"Unzipping -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

def main(unzip_archives: bool = True) -> None:
    for url, dest in DOWNLOADS:
        download(url, dest)

    if unzip_archives:
        for zp, out in UNZIP_TARGETS:
            unzip(zp, out)

    print("Done.")

if __name__ == "__main__":
    main(unzip_archives=True)
