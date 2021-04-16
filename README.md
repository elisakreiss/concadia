

# Concadia -- a corpus of **con**textualized images with **c**aptions and **a**lt **d**escriptions from Wikipe**dia**

Concadia is a corpus introduced in our paper "Concadia: Tackling image accessibility with context" (*to be submitted*) and contains Wikipedia images with their respective captions, alt descriptions and the broader context the images are situated in. We use this corpus to argue for a clear distinction between descriptions and captions, and show the similarities and differences between the two text forms. We further argue that captions and broader context are an important resource that can inform the generation of descriptions which are very sparse across the Web but absolutely crucial to make images accessible.

# How to create Concadia

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

