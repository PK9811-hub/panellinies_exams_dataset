import os
import json
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

def check_image_paths():
    env_data_path = os.getenv("DATA_DIR")
    if not env_data_path:
        logger.error ("Δε βρέθηκε το DATA_DIR στο .env.")
        return
    
    data_path = Path(env_data_path)
    
    missing_images = []
    total_images = 0
    checked_files = 0

    for json_path in data_path.rglob("*.json"):
        checked_files += 1
    
        with open (json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            continue
        
        for entry in data:
            images_list = entry.get("images", [])
            
        for image_dict in images_list:
            total_images += 1
            img_rel_path = image_dict.get("path", "")
            
            if img_rel_path == "":
                continue
        
            absolute_img_path = json_path.parent / img_rel_path
            
            if not absolute_img_path.exists():
                missing_images.append(
                    {"file": json_path.name,
                     "id": entry.get("id", "Άγνωστο ID"),
                     "path": img_rel_path}
                     )
                logger.warning(f"Δε βρέθηκε εικόνα για το αρχείο {json_path}.")

    print(f"\nΣυνολικά αρχεία JSON που ελέγχθηκαν: {checked_files}")
    print(f"Συνολικές αναφορές σε εικόνες που βρέθηκαν: {total_images}")

    if len(missing_images) > 0:
        logger.warning(f"ΠΡΟΣΟΧΗ! Βρέθηκαν {len(missing_images)} χαμένες εικόνες:")
    
        for img in missing_images:
            logger.warning(f" - Στο αρχείο {img['file']} (Ερώτηση {img['id']}): Έλειπε το {img['path']}")
    else:
        logger.info(f"Βρέθηκαν και οι {total_images} εικόνες.")

if __name__ == "__main__":
    check_image_paths()


    
