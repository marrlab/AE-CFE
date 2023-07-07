# We run this code for every dataset seperately 
from config import CellsConfig
import mrcnn.model_feat_extract as modellib
import os
import skimage
import numpy as np
import gzip
import pickle
import cv2
import sys

from tqdm import tqdm

#This is the path of the dataset that you want to extract the features from.
path = "/Matek-19/" 
width = 400
height = 400
#the output file
file_ext = 'Matek-19.dat.gz'
files = [f for f in os.listdir(path) if not f.startswith('.')]
config = CellsConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir='trained_model')
model.load_weights(model.find_last(), by_name=True)

features = {}
for file in files:
    images = [f for f in os.listdir(os.path.join(path,file)) if not f.startswith('.')]
    for image in tqdm(images):
        if image == '.':
            continue
        img = skimage.io.imread(os.path.join(path,file,image))
        img = cv2.resize(img, (width, height))
        if img.shape[-1] == 4:
            img = img[..., :3]
        output = model.detect([img])[0]
        image = image
        features[image] = {
            "label":file,
            "feats": output["feats"],
            "rois": output["rois"],
            "masks": output["masks"]
        }
print("Saving...")
with gzip.open(file_ext, "wb") as f:
    pickle.dump(features, f)
print("Done...")



