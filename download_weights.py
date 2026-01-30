import os
import zipfile
import urllib.request
import shutil

url = "https://ommer-lab.com/files/latent-diffusion/celeba.zip"
zip_path = "celeba.zip"
extract_dir = "celeba_tmp"
final_dir = "celeba"
final_ckpt_path = os.path.join(final_dir, "model.ckpt")

urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

ckpt_file = None
for root, _, files in os.walk(extract_dir):
    for f in files:
        if f.endswith(".ckpt"):
            ckpt_file = os.path.join(root, f)
            break
    if ckpt_file:
        break

if ckpt_file is None:
    raise FileNotFoundError("No .ckpt file found in the zip archive")

os.makedirs(final_dir, exist_ok=True)
shutil.move(ckpt_file, final_ckpt_path)

os.remove(zip_path)
shutil.rmtree(extract_dir)

print(f"Checkpoint saved to {final_ckpt_path}")
