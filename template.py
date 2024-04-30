import os
from pathlib import Path
import logging as log

log.basicConfig(level=log.INFO, format= '[%(asctime)s]: %(message)s:')

Project_name =  "Food_Classification"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{Project_name}/__init__.py",
    f"src/{Project_name}/ components/__init__.py",
    f"src/{Project_name}/utils/__init__.py",
    f"src/{Project_name}/config/__init__.py",
    f"src/{Project_name}/config/configuration.py",
    f"src/{Project_name}/entity/__init__.py",
    f"src/{Project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trails.ipynb",
    "templates/index.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir, filename = os.path.split(filepath)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        log.info(f'creating directory: {file_dir} for {filename}')

    if (not os.path.exists(filepath)) or os.path.getsize(filepath) == 0:
        with open(filepath,'w') as f:
            pass
        log.info(f'Creating empty directory: {filepath}')

    else:
        log.info('Direcotry already exists')


