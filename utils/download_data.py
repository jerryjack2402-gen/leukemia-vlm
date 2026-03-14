import gdown
import os
import zipfile

def download_dataset():
    # Google Drive folder link: https://drive.google.com/drive/folders/1_wPt38bPw4vh0gfDFuxpILlateERmxDA?usp=drive_link
    # Extracted ID from the link
    folder_id = "1ccYfuqaTDzcZcWdLL_jBv7k0f2mWVFpB"
    
    # We can't easily download a whole folder via gdown without it being a public link 
    # and using --folder flag, but often users prefer downloading a zip.
    # If it's a folder, we can try 'gdown --folder'
    
    output_dir = "data_path"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset from Google Drive folder ID: {folder_id}...")
    
    try:
        # gdown.download_folder shows a progress bar by default when quiet=False.
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
        
        # After download, list the files to "track" and verify
        print(f"\nScanning downloaded files in '{output_dir}'...")
        file_count = 0
        for root, dirs, files in os.walk(output_dir):
            file_count += len(files)
        
        print(f"\nSuccess! Dataset download complete.")
        print(f"Total files in '{output_dir}': {file_count}")
        print(f"Location: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\nError during download: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease ensure the Google Drive folder is shared with 'Anyone with the link'.")
        print(f"Manual link: https://drive.google.com/drive/folders/{folder_id}")

if __name__ == "__main__":
    download_dataset()
