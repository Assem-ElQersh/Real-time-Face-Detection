import os
import shutil
import cv2
import urllib.request
from pathlib import Path

class OpenCVSetup:
    def __init__(self):
        # Get the OpenCV data directory from the installation
        self.opencv_data_dir = os.path.join(os.path.dirname(cv2.__file__), 'data')
        if not os.path.exists(self.opencv_data_dir):
            # Try alternative path for some OpenCV installations
            self.opencv_data_dir = os.path.join(os.path.dirname(os.path.dirname(cv2.__file__)), 'data')
        
        self.target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'opencv')
        self.cascade_urls = {
            'haarcascade_frontalface_default.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
            'haarcascade_eye.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml'
        }
        
    def download_cascade_file(self, url, target_path):
        """Download the Haar Cascade file from OpenCV's GitHub repository"""
        try:
            print(f"Downloading Haar Cascade file from {url}")
            urllib.request.urlretrieve(url, target_path)
            return True
        except Exception as e:
            print(f"Error downloading Haar Cascade file: {str(e)}")
            return False
        
    def setup_opencv_files(self):
        """Copy required OpenCV files to the project directory"""
        try:
            # Create target directory
            os.makedirs(self.target_dir, exist_ok=True)
            
            # Set up Haar Cascade files
            for cascade_file, url in self.cascade_urls.items():
                source_path = os.path.join(self.opencv_data_dir, cascade_file)
                target_path = os.path.join(self.target_dir, cascade_file)
                
                if not os.path.exists(target_path):
                    # Try to copy from OpenCV installation first
                    if os.path.exists(source_path):
                        print(f"Copying {cascade_file} to {self.target_dir}")
                        shutil.copy2(source_path, target_path)
                    else:
                        # If not found locally, download from GitHub
                        if not self.download_cascade_file(url, target_path):
                            raise FileNotFoundError(f"Could not obtain {cascade_file}")
                
            # Set environment variable for OpenCV
            os.environ['OPENCV_DATA_PATH'] = self.target_dir
            
            return True
        except Exception as e:
            print(f"Error setting up OpenCV files: {str(e)}")
            return False

def main():
    setup = OpenCVSetup()
    if setup.setup_opencv_files():
        print("OpenCV setup completed successfully!")
    else:
        print("OpenCV setup failed!")

if __name__ == "__main__":
    main() 