import os
import shutil

def save_structured_file(block, compliance_id):
    base = f"data/structured/compliance_{compliance_id}"
    os.makedirs(base, exist_ok=True)

    src = block["value"]
    filename = os.path.basename(src)
    shutil.copy(src, f"{base}/{filename}")

    print(f"📊 Stored structured file: {filename}")
