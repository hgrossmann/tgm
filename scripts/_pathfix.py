from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_OF_PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PARENT_OF_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_OF_PROJECT_ROOT))
