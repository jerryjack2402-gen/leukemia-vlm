import gdown
import os
import zipfile

def download_dataset():
    # Google Drive folder link: https://drive.google.com/drive/folders/1_wPt38bPw4vh0gfDFuxpILlateERmxDA?usp=drive_link
    # Extracted ID from the link
    folder_id = "1_wPt38bPw4vh0gfDFuxpILlateERmxDA"
    
    # We can't easily download a whole folder via gdown without it being a public link 
    # and using --folder flag, but often users prefer downloading a zip.
    # If it's a folder, we can try 'gdown --folder'
    
    output_dir = "data_path"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset from Google Drive folder ID: {folder_id}...")
    
    try:
        # Note: gdown.download_folder exists but might require the folder to be shared correctly.
        # If it fails, we inform the user.
        gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
        print(f"\nSuccess! Dataset downloaded to '{output_dir}'.")
    except Exception as e:
        print(f"\nError downloading folder: {e}")
        print("Please ensure the Google Drive folder is shared with 'Anyone with the link' and permissions are set correctly.")
        print(f"Alternatively, visit: https://drive.google.com/drive/folders/{folder_id}")

if __name__ == "__main__":
    download_dataset()
