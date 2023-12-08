from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent
EXTERNAL_DIR = SCRIPT_DIR.joinpath(
    "../external")

sys.path.append(str(EXTERNAL_DIR.resolve()))
