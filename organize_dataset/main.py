import os
import shutil

#change path here if you want to run it yourself ^^
images_dir = r"G:\Users\Simon\Desktop\PROGRAMMERING\Datorseende\organize_dataset\all_images\images"
labels_dir = r"G:\Users\Simon\Desktop\PROGRAMMERING\Datorseende\organize_dataset\all_images\labels"

#where the organized pics will be saved
output_dir = r"G:\Users\Simon\Desktop\PROGRAMMERING\Datorseende\organize_dataset\sorted"

output_dirs = {
    "snowboard": os.path.join(output_dir, "snowboard"),
    "skier": os.path.join(output_dir, "skier")
}

#create directories
os.makedirs(output_dir, exist_ok=True)
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def get_unique_label(txt_filepath):
    labels = set()
    try:
        with open(txt_filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 1:
                    continue
                raw_label = parts[0]
                
                if raw_label.isdigit():
                    if raw_label == "0":
                        label = "snowboard"
                    elif raw_label == "1":
                        label = "skier"
                    else:
                        label = None
                else:
                    raw_label_lower = raw_label.lower()
                    if raw_label_lower in ["snowboarder", "snowboard"]:
                        label = "snowboard"
                    elif raw_label_lower == "skier":
                        label = "skier"
                    else:
                        label = None
                if label:
                    labels.add(label)
    except Exception as e:
        print(f"Kunde inte läsa {txt_filepath}: {e}")
        return None

    return labels.pop() if len(labels) == 1 else None

count_moved = 0
count_skipped = 0

for filename in os.listdir(images_dir):
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext in image_extensions:
        image_path = os.path.join(images_dir, filename)
        
        base_name = os.path.splitext(filename)[0]
        annotation_path = os.path.join(labels_dir, base_name + ".txt")

        if not os.path.exists(annotation_path):
            print(f"Ingen annoteringsfil hittades för {filename}, hoppar över.")
            count_skipped += 1
            continue

        label = get_unique_label(annotation_path)
        if label is None:
            print(f"{filename} har flera/ingen giltig etikett, hoppar över.")
            count_skipped += 1
            continue

        #move pic to correct folder
        dest_path = os.path.join(output_dirs[label], filename)
        shutil.move(image_path, dest_path)
        print(f"Flyttade {filename} till {output_dirs[label]}")
        count_moved += 1

print(f"Totalt flyttade bilder: {count_moved}")
print(f"Totalt hoppade över: {count_skipped}")
