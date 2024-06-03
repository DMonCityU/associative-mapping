import os
import re
import subprocess
import sys

def get_latest_script(script_base_name):
    script_pattern = re.compile(rf'{re.escape(script_base_name)}_v(\d+)(?:_(\d+))?(?:_(\d+))?\.py')
    latest_version = (0, 0, 0)
    latest_script = None

    for file in os.listdir('.'):
        match = script_pattern.match(file)
        if match:
            version_parts = match.groups()
            version = tuple(int(part) if part is not None else 0 for part in version_parts)
            if version > latest_version:
                latest_version = version
                latest_script = file

    return latest_script

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_latest.py <script_base_name> [additional_args...]")
        sys.exit(1)

    script_base_name = sys.argv[1]
    latest_script = get_latest_script(script_base_name)
    
    if latest_script:
        print(f"Running the latest script: {latest_script}")
        subprocess.run(['python', latest_script] + sys.argv[2:])
    else:
        print(f"No suitable script found for base name '{script_base_name}'.")

if __name__ == "__main__":
    main()
