from src.utils.model_downloader import ModelDownloader

if __name__ == "__main__":
    downloader = ModelDownloader()
    if downloader.download_model():
        print("Model setup completed successfully!")
    else:
        print("Model setup failed!") 