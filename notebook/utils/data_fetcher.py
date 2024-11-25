from tqdm import tqdm
import urllib.request
from pathlib import Path

# WANTED_SEASONS = ["0001", "0102", "0203", "0304", "0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
WANTED_SEASONS = ["0203", "0304", "0405", "0506", "0607", "0708", "0809", "0910", "1011", "1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
WANTED_LEAGUES = ["E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3", "D1", "D2", "I1", "I2", "SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"]

class DataFetcher():
    API_BASE_URL = "https://www.football-data.co.uk/mmz4281"
    
    def __init__(self, DATA_FOLDER_PATH=Path.cwd().joinpath("..","data")):
        self.DATA_FOLDER_PATH = DATA_FOLDER_PATH

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    def download_url(self, download_path: Path, url: str):
        with self.DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=download_path, reporthook=t.update_to)

    def download_dataset(self, download_path: Path, url: str):
        print("Downloading dataset...")
        self.download_url(url=url, download_path=download_path)
        print("Download complete!")

    def main(self):
        if not self.DATA_FOLDER_PATH.exists():
            self.DATA_FOLDER_PATH.mkdir(parents=True)

        for league in WANTED_LEAGUES:
            for season in WANTED_SEASONS:
                request_url = f"{self.API_BASE_URL}/{season}/{league}"

                download_path = self.DATA_FOLDER_PATH.joinpath(league)

                if not download_path.exists():
                    download_path.mkdir(parents=True)
                
                download_path = download_path.joinpath(f"{league}-{season}.csv")

                if download_path.exists():
                    continue

                try:
                    self.download_dataset(download_path=download_path, url=request_url)    
                except:
                    print(f"{request_url} does not exist.")

if __name__ == "__main__":
    data_fetcher = DataFetcher(DATADATA_FOLDER_PATH=Path.cwd().joinpath("data"))
    data_fetcher.main()