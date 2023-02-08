import torch
import os
from tqdm import tqdm
from Dataloader import DataLoader
import torch.nn as nn
import pickle
import gzip
import sys
import torch




file = "AE_CFE.dat"
model_name = "AE_GB_MMD20220227-003348.mdl"

equivalent_classes = {

    #Acevedo-20 dataset
    'basophil': 'basophil',
    'eosinophil': 'eosinophil',
    'erythroblast': 'erythroblast',
    'IG': "unknown", #immature granulocytes,
    'PMY': 'promyelocyte', #immature granulocytes,
    'MY': 'myelocyte', #immature granulocytes,
    'MMY': 'metamyelocyte', #immature granulocytes,
    'lymphocyte': 'lymphocyte_typical',
    'monocyte': 'monocyte',
    'NEUTROPHIL': "unknown",
    'BNE': 'neutrophil_banded',
    'SNE': 'neutrophil_segmented',
    'platelet': "unknown",
    #Matek-19 dataset
    'BAS': 'basophil',
    'EBO': 'erythroblast',
    'EOS': 'eosinophil',
    'KSC': 'smudge_cell',
    'LYA': 'lymphocyte_atypical',
    'LYT': 'lymphocyte_typical',
    'MMZ': 'metamyelocyte',
    'MOB': 'monocyte', #monoblast
    'MON': 'monocyte',
    'MYB': 'myelocyte',
    'MYO': 'myeloblast',
    'NGB': 'neutrophil_banded',
    'NGS': 'neutrophil_segmented',
    'PMB': "unknown",
    'PMO': 'promyelocyte',
    #INT-20 dataset
    '01-NORMO': 'erythroblast',
    '04-LGL': "unknown", #atypical
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
        'myeloblast' : 3,
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



ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model= torch.load(os.path.join("Model/",model_name ), torch.device('cpu'))

model.eval()



if ngpu > 1:
    encoder_model = nn.DataParallel(model)
    model = model.to(device)
else:
    model = model.to(device)

torch.cuda.empty_cache()
## Load the dataset
dataset = DataLoader()
traindataloader = torch.utils.data.DataLoader(dataset, batch_size=max(20 * ngpu,2), shuffle=True)  ##32


features =[]

for (feat, scimg, label, db,key) in tqdm(traindataloader, desc='images'):
    feat = feat.squeeze()
    feat = feat.float()
    feat = feat.to(device)

    z, _, _ = model(feat)
    z = z.cpu().detach().numpy()
    datasets_names = ["Matek-19", "INT-20","Acevedo-20"]


    for i in range(len(z)):
        features.append({"z": z[i], "label": label_map[equivalent_classes[label[i]]], "dataset": db[i].cpu().detach().numpy().argmax(),"key":key})


print("saving...")

if os.path.exists(os.path.join('Features_Files/')) is False:
    os.makedirs(os.path.join('Features_Files/'))
with gzip.open(os.path.join('Features_Files/', file), "wb") as f:
    pickle.dump([features], f)
