import os
import requests
import bz2
import shutil
import sys

def download_file(url, filename):
    """Download a file from URL using requests"""
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Get total size if available
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                f.write(data)
                # Only show progress if we know the total size
                if total_size > 0:
                    done = int(50 * downloaded / total_size)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\rDownloaded: {downloaded} bytes")
                    sys.stdout.flush()
        print("\nDownload completed!")
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading file: {e}")
        return False
    return True

def extract_bz2(filename):
    """Extract a bz2 file"""
    print(f"Extracting {filename}...")
    try:
        with bz2.open(filename, 'rb') as source, open(filename[:-4], 'wb') as dest:
            shutil.copyfileobj(source, dest)
        os.remove(filename)
        print(f"Extracted {filename}")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Download face detection model
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = os.path.join('models', 'shape_predictor_68_face_landmarks.dat.bz2')
    
    # Alternative download URL (mirror)
    mirror_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    
    print("Attempting to download face detection model...")
    
    # Try primary URL first
    if not download_file(model_url, model_path):
        print("\nTrying alternative download URL...")
        if not download_file(mirror_url, model_path):
            print("\nBoth download attempts failed. Please try the following alternatives:")
            print("1. Download manually from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2")
            print("2. Place the downloaded file in the 'models' directory")
            print("3. Run this script again to extract the file")
            return
    
    if extract_bz2(model_path):
        print("Model download and extraction completed successfully!")
    else:
        print("Error: Failed to extract the model file")

if __name__ == "__main__":
    main() 