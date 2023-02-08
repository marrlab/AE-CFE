import gzip
import numpy as np
import pickle
import os
import cv2
from torch.utils.data import Dataset

equivalent_classes = {

    # Acevedo-20 dataset
    'basophil': 'basophil',
    'eosinophil': 'eosinophil',
    'erythroblast': 'erythroblast',
    'IG': "unknown",  # immature granulocytes,
    'PMY': 'promyelocyte',  # immature granulocytes,
    'MY': 'myelocyte',  # immature granulocytes,
    'MMY': 'metamyelocyte',  # immature granulocytes,
    'lymphocyte': 'lymphocyte_typical',
    'monocyte': 'monocyte',
    'NEUTROPHIL': "unknown",
    'BNE': 'neutrophil_banded',
    'SNE': 'neutrophil_segmented',
    'platelet': "unknown",
    # Matek-19 dataset
    'BAS': 'basophil',
    'EBO': 'erythroblast',
    'EOS': 'eosinophil',
    'KSC': 'smudge_cell',
    'LYA': 'lymphocyte_atypical',
    'LYT': 'lymphocyte_typical',
    'MMZ': 'metamyelocyte',
    'MOB': 'monocyte',  # monoblast
    'MON': 'monocyte',
    'MYB': 'myelocyte',
    'MYO': 'myeloblast',
    'NGB': 'neutrophil_banded',
    'NGS': 'neutrophil_segmented',
    'PMB': "unknown",
    'PMO': 'promyelocyte',
    #  INT-20 dataset
    '01-NORMO': 'erythroblast',
    '04-LGL': "unknown",  # atypical
    '05-MONO': 'monocyte',
    '08-LYMPH-neo': 'lymphocyte_atypical',
    '09-BASO': 'basophil',
    '10-EOS': 'eosinophil',
    '11-STAB': 'neutrophil_banded',
    '12-LYMPH-reaktiv': 'lymphocyte_atypical',
    '13-MYBL': 'myeloblast',
    '14-LYMPH-typ': 'lymphocyte_typical',
    '15-SEG': 'neutrophil_segmented',
    '16-PLZ': "unknown",
    '17-Kernschatten': 'smudge_cell',
    '18-PMYEL': 'promyelocyte',
    '19-MYEL': 'myelocyte',
    '20-Meta': 'metamyelocyte',
    '21-Haarzelle': "unknown",
    '22-Atyp-PMYEL': "unknown",
}

label_map = {
    'basophil': 0,
    'eosinophil': 1,
    'erythroblast': 2,
    'myeloblast': 3,
    'promyelocyte': 4,
    'myelocyte': 5,
    'metamyelocyte': 6,
    'neutrophil_banded': 7,
    'neutrophil_segmented': 8,
    'monocyte': 9,
    'lymphocyte_typical': 10,
    'lymphocyte_atypical': 11,
    'smudge_cell': 12,
}
class DataLoader(Dataset):
    def __init__(self):
        self.datasets_names = ["Matek-19", "INT-20", "Acevedo-20"]
       
        # loading features
        datasets = {}
        remove_keys = ["15-48904.PB.PAP~B.1272-1272.TIF", "15-48904.PB.PAP~B.1514-1514.TIF"]
        
        datasets_dir = "/lustre/groups/aih/raheleh.salehi/Aug_features_datasets/"
        keys = np.unique([x.split("-")[1] for x in os.listdir(datasets_dir)])

        for k in keys:
            datasets[k] = [x for x in os.listdir(datasets_dir) if x.split("-")[1] == k]

        samples = {}
        for dataset in datasets:
            for file in datasets[dataset]:
                print("loading ", dataset, "... ", end="", flush=True)
                with gzip.open(os.path.join(datasets_dir, file), "rb") as f:
                    data = pickle.load(f)
                    for d in data:
                        data[d]["dataset"] = dataset
                        if "label" not in data[d].keys() and dataset == "AML":
                            data[d]["label"] = d.split("_")[0]
                    samples = {**samples, **data}
                print("[done]")

        samples2 = samples.copy()
        for s in samples2:
            if equivalent_classes[samples[s]["label"]] == "unknown":
                samples.pop(s, None)

        for k in remove_keys:
            samples.pop(k, None)
        # for k in cleanup_pbc:
        #     samples.pop(k, None)
            
        # loading images
        images = {}
        images_dir = "/lustre/groups/aih/raheleh.salehi/save_files/"
        image_files = os.listdir(images_dir)
        for img_file in image_files:
            print("loading", img_file, "...", end="", flush=True)
            with gzip.open(os.path.join(images_dir, img_file), "rb") as f:
                file_images = pickle.load(f)
            images = {**images, **file_images}
            print("[done]")

        self.datasets = datasets
        self.samples = samples
        self.images = images

        self.data = list(set(self.samples.keys()) & set(self.images.keys()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key = self.data[index]
        label_fold = self.samples[key]['label']
        img = self.images[key]
        bounding_box = self.samples[key]['rois']
        if len(bounding_box) == 1:
            bounding_box = bounding_box[0]
        w,h, _ = img.shape
        bounding_box = bounding_box / 400
        x0 = bounding_box[0] * w
        y0 = bounding_box[1] * h
        x1 = bounding_box[2] * w
        y1 = bounding_box[3] * h

        roi_cropped = img[max(0, int(x0) - 10):min(w, int(x1) + 20), max(0, int(y0) - 10):min(h, int(y1) + 20)]
        roi_cropped = cv2.resize(roi_cropped, (128, 128))
        roi_cropped = roi_cropped / 255.
       
        roi_cropped = np.rollaxis(roi_cropped, 2, 0)
        feat = self.samples[key]['feats']
        feat = 2. * (feat - np.min(feat)) / np.ptp(feat) - 1
        feat = np.squeeze(feat)
        feat = np.rollaxis(feat, 2, 0)

        ds = np.zeros(len(self.datasets_names))
        ds[self.datasets_names.index(self.samples[key]['dataset'])] = 1




        return feat, roi_cropped, label_fold, ds,key
       

