import os
from pathlib import Path
import logging 


logging.basicConfig(level=logging.INFO, format="[INFO][%(asctime)s]: %(message)s:")


project_name = "jobClassification"

file_lists = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",  
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/__init__.py"
]

for fp in file_lists:
    fp = Path(fp)
    dir, fname = os.path.split(fp)
    if dir != "":
        os.makedirs(dir,exist_ok=True)
        logging.info(f"Created directory: {dir} for {fname}")
    if (not os.path.exists(fp)) or (os.path.getsize(fp) == 0):
        with open(fp, "w") as f:
            logging.info(f"Created empty file {fname}")
            pass
    else:
        logging.info(f"{fname} already exists!!")
 
            
