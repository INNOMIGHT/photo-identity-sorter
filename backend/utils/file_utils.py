import os
import shutil
from zipfile import ZipFile

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def save_grouped_photos(groups, output_dir):
    clear_folder(output_dir)

    for person, photos in groups.items():
        person_folder = os.path.join(output_dir, person)
        os.makedirs(person_folder, exist_ok=True)

        for photo in photos:
            filename = os.path.basename(photo)
            shutil.copy(photo, os.path.join(person_folder, filename))

    zip_path = os.path.join(output_dir, "result.zip")
    with ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file != "result.zip":
                    full_path = os.path.join(root, file)
                    zipf.write(full_path, os.path.relpath(full_path, output_dir))

    return zip_path
