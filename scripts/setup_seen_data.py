import os, json, shutil
from pathlib import Path

def main():
    # Read seen index
    seen_items = json.load(open("data/seen/index.json"))
    
    # Create seen_data structure
    seen_data = Path("data/seen_data")
    clips_dir = seen_data / "clips"
    seen_data.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy only the clips referenced in seen index
    for item in seen_items:
        src_path = Path("data/mini") / item["path"]
        dst_path = clips_dir / Path(item["path"]).name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # Copy the seen index
    shutil.copy2("data/seen/index.json", seen_data / "index.json")
    
    print(f"[OK] Created seen_data with {len(seen_items)} clips")

if __name__ == "__main__":
    main()
