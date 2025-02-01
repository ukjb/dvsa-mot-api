import requests
import os
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    def __init__(self):
        self.url = "https://assets.publishing.service.gov.uk/media/66f15c9b34de29965b489bd2/df_VEH0120_GB.csv"
        # Get the directory where the script is located
        self.script_dir = Path(__file__).parent.absolute()
        self.dataset_path = self.script_dir / "vehicle_statistics.csv"
        
    def verify_csv(self, file_path):
        """Verify that the CSV file is valid and contains expected data"""
        try:
            df = pd.read_csv(file_path)
            # Check if dataframe has content
            if df.empty:
                return False, "CSV file is empty"
            
            # Check for required columns
            required_columns = ['Make', 'Model']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            return True, f"CSV verified successfully. Shape: {df.shape}"
            
        except Exception as e:
            return False, f"CSV verification failed: {str(e)}"

    def download_dataset(self):
        """Download the vehicle statistics dataset"""
        try:
            logger.info(f"Downloading dataset from {self.url}")
            
            # Create backup of existing file if it exists
            if self.dataset_path.exists():
                backup_path = self.dataset_path.with_suffix('.csv.backup')
                self.dataset_path.rename(backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            # Download new file
            response = requests.get(self.url, timeout=30)
            response.raise_for_status()
            
            # Save the file
            with open(self.dataset_path, "wb") as file:
                file.write(response.content)
            
            # Verify the downloaded file
            is_valid, message = self.verify_csv(self.dataset_path)
            
            if is_valid:
                logger.info(f"Dataset downloaded and verified successfully: {message}")
                if 'backup_path' in locals():
                    backup_path.unlink()  # Remove backup if verification successful
                return True
            else:
                logger.error(f"Downloaded file verification failed: {message}")
                if 'backup_path' in locals():
                    # Restore backup
                    backup_path.rename(self.dataset_path)
                    logger.info("Restored backup file")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def get_file_info(self):
        """Get information about the current dataset file"""
        try:
            if not self.dataset_path.exists():
                return "No dataset file found"
            
            file_stats = self.dataset_path.stat()
            file_size = file_stats.st_size / (1024 * 1024)  # Convert to MB
            mod_time = datetime.fromtimestamp(file_stats.st_mtime)
            
            df = pd.read_csv(self.dataset_path)
            
            return {
                "file_path": str(self.dataset_path),
                "file_size": f"{file_size:.2f} MB",
                "last_modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                "rows": len(df),
                "columns": len(df.columns)
            }
            
        except Exception as e:
            return f"Error getting file info: {e}"

def main():
    downloader = DatasetDownloader()
    
    # Download the dataset
    success = downloader.download_dataset()
    
    if success:
        # Print file information
        file_info = downloader.get_file_info()
        logger.info("Dataset file information:")
        for key, value in file_info.items():
            logger.info(f"{key}: {value}")

if __name__ == "__main__":
    main()
