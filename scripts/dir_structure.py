import os
import json
from datetime import datetime


def generate_directory_structure(startpath):
    # For storing the JSON structure
    structure_dict = {}
    # For storing the MD structure
    md_content = [
        "# Project Directory Structure",
        f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    def process_directory(path, indent="", current_dict=None):
        if current_dict is None:
            current_dict = structure_dict

        # For MD file
        md_content.append(f"\n{indent}📁 {os.path.basename(path)}")

        # For JSON structure
        current_dict["type"] = "directory"
        current_dict["name"] = os.path.basename(path)
        current_dict["contents"] = []

        ignore_dirs = {".git", "venv", "__pycache__", "node_modules"}

        for entry in os.scandir(path):
            if entry.name in ignore_dirs:
                continue

            if entry.is_file():
                # Add to MD
                md_content.append(f"{indent}  📄 {entry.name}")
                # Add to JSON
                current_dict["contents"].append({"type": "file", "name": entry.name})
            elif entry.is_dir():
                # Add to JSON
                new_dict = {}
                current_dict["contents"].append(new_dict)
                process_directory(entry.path, indent + "  ", new_dict)

    process_directory(startpath)

    # Save to MD file
    with open("directory_structure.md", "w", encoding="utf-8") as md_file:
        md_file.write("\n".join(md_content))

    # Save to JSON file
    with open("directory_structure.json", "w", encoding="utf-8") as json_file:
        json.dump(structure_dict, json_file, indent=2)

    return "Directory structure has been saved to 'directory_structure.md' and 'directory_structure.json'"


print(generate_directory_structure("../"))