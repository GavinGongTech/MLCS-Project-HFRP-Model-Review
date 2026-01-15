# final_project/download_filings.py

# Buffer Code from before; did not use this for final implementation
# But good to have for reference as scrap code
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EDGAR_INTERNAL = PROJECT_ROOT / "edgar_crawler"

for p in (PROJECT_ROOT, EDGAR_INTERNAL):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from edgar_crawler.download_filings import main as edgar_download_main
import edgar_crawler.extract_items as extract_items_module  # or main, depending on that file

def main():
    edgar_download_main()

    extract_items_module.main()

if __name__ == "__main__":
    main()
