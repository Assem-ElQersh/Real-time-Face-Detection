import os
import bz2
import urllib.request
import shutil
from tqdm import tqdm

class ModelDownloader:
    def __init__(self):
        self.model_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models")
        self.model_path = os.path.join(self.models_dir, "shape_predictor_68_face_landmarks.dat")
        
    def download_with_progress(self, url, filename):
        """Download a file with a progress bar"""
        with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
            total_size = int(response.info().get('Content-Length', 0))
            block_size = 1024 * 1024  # 1MB
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    pbar.update(len(buffer))
    
    def download_model(self):
        """Download and extract the face alignment model"""
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Check if model already exists
        if os.path.exists(self.model_path):
            print(f"Model already exists at {self.model_path}")
            return True
            
        # Download the compressed model
        temp_file = self.model_path + ".bz2"
        print(f"Downloading model from {self.model_url}")
        try:
            self.download_with_progress(self.model_url, temp_file)
        except Exception as e:
            print(f"Error downloading model: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False
            
        # Extract the model
        print("Extracting model...")
        try:
            with bz2.open(temp_file, 'rb') as source, open(self.model_path, 'wb') as dest:
                shutil.copyfileobj(source, dest)
        except Exception as e:
            print(f"Error extracting model: {str(e)}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return False
            
        # Clean up
        os.remove(temp_file)
        print(f"Model downloaded and extracted to {self.model_path}")
        return True

def main():
    downloader = ModelDownloader()
    if downloader.download_model():
        print("Model setup completed successfully!")
    else:
        print("Model setup failed!")

if __name__ == "__main__":
    main() 