import os
import glob
# import requests
from utils import display_many, download_images,  get_urls, 
# from utils import load_image, resize_with_pad


# Define location to download files to
scrape_dir = "google_scrape/downloaded"
if not os.path.exists(scrape_dir):
    os.makedirs(scrape_dir)     # Breaks if parent folder does not exist

# Query term
query = "young woman"
query = '_'.join(query.split(' '))

# Read the list of urls scraped from Google
url_path = 'google_scrape/urls/urls.txt'
urls = get_urls(url_path)

# Download images from list
download_images(urls, dir=scrape_dir, label=query)

# Load local paths to downloaded images
img_paths = glob.glob("{}/{}_[0-9]*.jpg".format(scrape_dir, query))

# Remove corrupt images
print("Removing corrupt files...")
for i in range(len(img_paths)):
    try:
        img = load_image(img_paths[i])
    except:
        os.remove(img_paths[i])

# Load local paths to downloaded images
img_paths = glob.glob("{}/{}_[0-9]*.jpg".format(scrape_dir, query))

print("{} images available.".format(len(img_paths)))

# Display the first 25 downloaded images
display_many(img_paths)