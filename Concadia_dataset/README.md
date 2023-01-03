

# CONCADIA -- a dataset of CONtextualized images with Captions and Alt Descriptions from WikipedIA

## Download Concadia

The Concadia dataset requires the following two resources: 

[Click here](https://drive.google.com/file/d/1gDhVlOwcGcwBT5LWYwgn9xEElGlKVpFb/view?usp=sharing) to download `resized.zip`, which contains all images already sized to 256x256px each.

[Click here](https://drive.google.com/file/d/1kiTSiqk7y7JdHssXjoLwcOomC7lhb5k8/view?usp=sharing) to download `wiki_split.json`, which contains the image names and sources, their corresponding descriptions, captions, and immediate paragraph information. It also already contains a train/val/test split. The file format is very similar to the Karpathy test splits for MS-COCO.

Please, contact me at ekreiss@stanford.edu if there are any issues with the above files or you're experiencing any issues with using this dataset.

## Experimental Data

Participants rated original and model-generated descriptions and captions for 300 images according to how well the text could replace the image and how much one could learn from the text that couldn't be learned from the image. The data is available [here](https://drive.google.com/file/d/1o8dFFafLdYFeIm6BYvM7C0r9VmFD256F/view?usp=sharing).

## How to create Concadia from sratch

Crawling all Wikipedia articles is very time consuming. Instead, we parse the publicly available Wikipedia XML file to find articles with all potential datapoints, i.e., that contain images with alt descriptions and captions. Then, we only have to crawl the articles with potential datapoints and extract the file path, alt descriptions, captions and accompanying contexts.

Note that currently the file paths are still matched to my personal file storage which needs to be changed before running the scripts.

1) Download English Wikipedia multistream xml file and save it to your data directory.

2) Manually change the ```xml``` file ending to ```txt```.

3) Run ```python find_articles.py``` to extract all Wikipedia articles with potential datapoints which get saved in ```articleextr.csv```.

4) Crawl promising articles from Wikipedia and extract images, alt descriptions, contexts and accompanying paragraphs by running ```python crawl_labels.py```. The results are saved to ```fileextr_crawl.csv```.

5) ```python crawl_images.py``` takes in ```fileextr_crawl.csv```. It crawls images, saves them in ```wikicommons/wikiimages_raw``` and saves all data + filenames in ```data_wfilenames.csv```.

6) ```resize.py``` takes the images saved in ```wikicommons/wikiimages_raw``` and saves the resized images in ```wikicommons/resized```.

7) Run ```create_split_json.py``` to clean the data and create a challenging train/val/test split. Needs ```data_wfilenames.csv``` as input and access to resized images and outputs ```wiki_split.json```.

8) ```json_postprocess.py``` creates a csv file (```final_data.csv```) from ```wiki_split.json``` for subsequent analyses and creates csv and json files (of the form ```final_data_random``` + scrambled_column + ```.csv```) for the scrambled baselines.

optional) ```crawl_copyrightinfo.py``` uses ```data_wfilenames.csv``` to crawl copyright information for each image and saves the data in ```data_wcopyright.csv```.

