import shutil
import os
from os.path import join

def _clear_log():
    print('\r' + ' ' * shutil.get_terminal_size().columns, end='\r')

def log_dynamic(message: str):
    _clear_log()
    print(message, end='\r', flush=True)

def log(message: str):
    _clear_log()
    print(message)

def copy_original_images(tab: dict[str, list[str]], result_directory: str):
    for _class, files in tab.items():
        new_directory = join(result_directory, _class)
        os.makedirs(new_directory, exist_ok=True)

        for file in files:
            dest_path = join(new_directory, os.path.basename(file))
            shutil.copyfile(file, dest_path)
