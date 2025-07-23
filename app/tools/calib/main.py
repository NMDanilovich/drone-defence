from pathlib import Path
import shutil

# --- find images ---
IMAGES_PATH = Path("/root/yandex_disk/dataset/Dataset_drones/train/images")
images = IMAGES_PATH.glob("*.jpg")


# --- make directory ---
CALIB_SET = Path("./images")
CALIB_SET.mkdir(exist_ok=True)


# --- copy files ---
for i, path in enumerate(images):
    if i == 200:
        break
    
    new_name = str(i).zfill(5) + ".jpg" # example: 002.jpg 
    shutil.copyfile(path, CALIB_SET/new_name)


# --- write paths in text file ---
# text_file = CALIB_SET / "data.txt"
# with open(text_file, "w") as file:
#     line_generator = (f"{path}\n" for path in CALIB_SET.glob("*.jpg"))
#     file.writelines(line_generator)

# --- write paths in yaml file ---
# without yaml library...
n_classes = 6
text_file = CALIB_SET / "data.yaml"
with open(text_file, "w") as file:
    file.write("train: ./")
    file.write("val: ./")
    file.write("test: ./")
    file.write(f"nc: {n_classes}")
    