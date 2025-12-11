import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
from urllib.parse import urljoin
from urllib.request import urlretrieve

# === CONFIG ===
INDEX_URL = "https://cis.whoi.edu/science/B/whalesounds/index.cfm"
OUTPUT_DIR = "dataset"
MAX_FILES_PER_SPECIES = 1000
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# === GET SPECIES LIST ===
def get_species_list():
    print("Fetching species list...")
    r = requests.get(INDEX_URL, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    select_box = soup.find("select", {"id": "getSpeciesCommon"})
    if not select_box:
        raise RuntimeError("Species <select> not found on index.cfm")

    species_list = []
    for option in select_box.find_all("option")[1:]:  # skip "Select"
        url = option["value"].strip()
        name = option.text.strip()
        species_list.append((name, url))

    print(f"Found {len(species_list)} species.")
    return species_list

# === SCRAPE METADATA FOR A FILE ===
def get_wav_metadata(rn_id):
    meta_url = urljoin(INDEX_URL, f"metaData.cfm?RN={rn_id}")
    r = requests.get(meta_url, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    metadata = {}
    # Only parse rows with exactly 2 <td> elements
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) == 2:
            key = tds[0].text.strip().replace(":", "")
            value = tds[1].text.strip()
            if key and value:  # Skip empty keys/values
                metadata[key] = value
    return metadata

# === DOWNLOAD WAV FILES + METADATA ===
def download_species_bestof(name, url):
    print(f"\n=== {name} ===")
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    rows = soup.find_all("tr")
    folder_name = name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    species_dir = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(species_dir, exist_ok=True)

    metadata_list = []

    for i, tr in enumerate(rows, start=1):
        wav_link_tag = tr.find("a", href=lambda x: x and x.lower().endswith(".wav"))
        if not wav_link_tag:
            continue
        file_url = urljoin(url, wav_link_tag["href"])
        filename = os.path.basename(file_url)
        save_path = os.path.join(species_dir, filename)

        meta_tag = tr.find("a", href=lambda x: x and "popUpWin" in x)
        rn_id = None
        if meta_tag:
            rn_id = meta_tag["href"].split("'")[1].split("=")[1].split(")")[0]

        # Download WAV if it doesn't exist
        if not os.path.exists(save_path):
            print(f"Downloading {i}/{len(rows)}: {filename}")
            try:
                urlretrieve(file_url, save_path)
            except Exception as e:
                print(f"Error downloading {file_url}: {e}")
                continue

        # Get metadata
        file_metadata = {"species": name, "file": filename, "url": file_url, "local_path": save_path}
        if rn_id:
            try:
                file_metadata.update(get_wav_metadata(rn_id))
            except Exception as e:
                print(f"Error fetching metadata for RN {rn_id}: {e}")

        metadata_list.append(file_metadata)

        if len(metadata_list) >= MAX_FILES_PER_SPECIES:
            break

    # Save metadata CSV
    pd.DataFrame(metadata_list).to_csv(os.path.join(species_dir, "metadata.csv"), index=False)
    print(f"Metadata saved for {name}")

# === MAIN ===
def main():
    species_list = get_species_list()
    for name, url in species_list:
        download_species_bestof(name, url)

if __name__ == "__main__":
    main()