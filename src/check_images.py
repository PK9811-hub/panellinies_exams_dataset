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
        logger.error ("DATA_DIR not found in .env.")
        return
    
    data_path = Path(env_data_path)
    
    missing_images = []
    total_images = 0
    checked_files = 0
    total_ids_checked = 0

    for json_path in data_path.rglob("*.json"):
        if "GEL" not in json_path.parts:
            continue
        checked_files += 1
    
        with open (json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            continue
        
        for entry in data:
            total_ids_checked += 1
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
                        "id": entry.get("id", "Unknown ID"),
                        "path": img_rel_path}
                        )
                    logger.warning(f"Image not found for file {json_path}.")

    print(f"\nTotal JSON files checked: {checked_files}")
    print(f"Total IDs (questions) checked: {total_ids_checked}")

    if len(missing_images) > 0:
        logger.warning(f"WARNING! Found {len(missing_images)} missing images:")
    
        for img in missing_images:
            logger.warning(f" - In file {img['file']} (Question {img['id']}): Missing {img['path']}")
    else:
        logger.info(f"Found all {total_images} images.")

if __name__ == "__main__":
    check_image_paths()


    
