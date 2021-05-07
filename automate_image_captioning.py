# -*- coding: utf-8 -*-
"""anomaly_caption_labelling.ipynb


Data Preperation

"""Read your image file paths to a list."""

import os
root_dir = '/content/drive/MyDrive/anomaly_captioning/dataset/test/images/test'
image_list = []
for subdir, dirs, files in os.walk(root_dir):
  for file in files:
    image_list.append(subdir+ '/'+file)
image_list.sort()

"""Create a simple comma seperated file to save;

*   *image path,* to read and display alter
*   *image label status*, (1:labelled, 2:not labelled)
*   *caption*, generated for the particular image


Save this file to either your drive or local. Preffered practice is to save it to drive and make it available to use repetetivly as image captioning is not a one-shot study. You need to spend hours. So you may need to run this script again. 
"""

# CREATE A CSV FOR STORING INFORMATION
label_stat = [0 for i in image_list] # 0 not labelled, 1 labelled, all 0 at the beginning
caption = [0 for i in image_list]
d = {'image':image_list,'label_stat':label_stat, 'caption': caption}

df = pd.DataFrame(d)
df.head(5)
df.to_csv('/content/drive/MyDrive/data.csv')

"""Mount your google drive to use your last updated image captions dataset. """

# mount it
from google.colab import drive
drive.mount('/content/drive')

"""
Modules required to import: 
"""

# Commented out IPython magic to ensure Python compatibility.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv 
import matplotlib.pylab as plt 
import time
from IPython.display import clear_output
from time import sleep
# %matplotlib inline

"""You can press E to exit so that you can continue captioning whenever you want. Your changes will be saved to df_updated. Next time you can keep captioning where you left off!"""

def caption_image(df):
  exit = 'c'
  while exit != 'e':
    for idx, row in df.iterrows():
        if df.loc[idx, 'label_stat'] == 0:
            print(str(idx+1),'. image is going to be captioned!')
            image_path = df.loc[idx, 'image']
            img = cv2.imread(image_path) 
            plt.imshow(img)     
            plt.show()
            sleep(1)
            caption_sentence = input('Please enter the caption for this image: ')
            df.loc[idx, 'caption'] = caption_sentence
            df.loc[idx, 'label_stat'] = 1
            print('caption value is set...')
            exit = input('Press E to exit , any other to contiunie!')
            clear_output(wait=True)
            if exit == 'e':
              print(exit)
              print('Caption operation has been stopped.')
              break
    exit = 'e'
  return df

"""Create an updated file and save it. """

df = pd.read_csv('/content/drive/MyDrive/dataset/data.csv')
df_updated = caption_image(df)

df_updated.to_csv('/content/drive/MyDrive/dataset/captioned_data.csv')
