# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:41:55 2019

@author: Emmanuel Vezzoli
"""

# Import des librairies
import os
import random
import glob
import shutil
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Téléchargement des données
Kaggle_username = "evezzoli"
Kaggle_key = "ff67ecd62784ca247c34335615fd77b3"
os.environ['KAGGLE_USERNAME'] = Kaggle_username
os.environ['KAGGLE_KEY'] = Kaggle_key
# !kaggle competitions download -c pku-autonomous-driving

# Création des fonctions
# Création d'un dossier


def create_repertory(repo):
    """
    fonction de création de dossier
    """
    try:
        os.mkdir(repo)
    except:
        print('Le dossier est existant')


# Affichage d'une image et son histogramme

def print_image(img):
    """
    fonction d'affichage de l'image
    """
    # On affiche l'image
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    # On affiche l'histogramme
    plt.subplot(1, 2, 2)
    plt.hist(img.flatten(), bins=range(256))
    plt.show()


# Reformater lune image
def image_format(index, border, size):
    image1 = Image.open(train['train_img'][index])
    image1 = ImageOps.expand(image1, border=border, fill=0)
    x1 = np.array(image1).shape[0]
    y1 = np.array(image1).shape[1]
    image1 = ImageOps.fit(image1, (max(x1, y1), max(x1, y1)),
                          2, 0.0, (0.5, 0.5))
    image1 = image1.resize((size, size), resample=0)
    return image1

# Prepare data
def prepare_data(alpha=0.2, nb_pics=0.2, size=128, border=0):
    # On supprime les dossiers existants
    shutil.rmtree('Data/Train', True)
    shutil.rmtree('Data/Test', True)
    shutil.rmtree('Data/Val', True)
    # On créé les dossier d'entrainement
    create_repertory('Data/Train')
    # On créé les dossier de test
    create_repertory('Data/Test')
    # On créé les dossier de test
    create_repertory('Data/Val')
    liste_index_train = []
    nb_pic = int(len(list(train['train_img'].index))*nb_pics)
    liste_index_train = random.sample(list(train['train_img'].index), nb_pic)
    a = int(len(liste_index_train)*alpha)
    liste_index_test = random.sample(liste_index_train, a)
    for j in liste_index_test:
        del liste_index_train[liste_index_train.index(j)]
    liste_index_val = random.sample(liste_index_train, a)
    for j in liste_index_val:
        del liste_index_train[liste_index_val.index(j)]
    # création du jeu d'entrainement
    for k in liste_index_train:
        img = image_format(k, border, size)
        img = img.save('Data/Train/' + train['ImageId'][k] + '.jpg')
    # création du jeu de test
    for k in liste_index_test:
        img = image_format(k, border, size)
        img = img.save('Data/Test/' + train['ImageId'][k] + '.jpg')
    # création du jeu de val
    for k in liste_index_val:
        img = image_format(k, border, size)
        img = img.save('Data/Val/' + train['ImageId'][k] + '.jpg')
    return liste_index_train, liste_index_test, liste_index_val


# Décompression des fichiers
# On liste les fichiers
files = glob.glob("/content/*.zip")
# On supprime le dossier
shutil.rmtree('Data', True)
# On créé les dossier de données
create_repertory('Data')
# On stocke le dossier de travail dans la variable path
path = os.getcwd()

# ouvrir les fichiers zip en mode lecture
for file in files:
    with ZipFile(file, 'r') as zip1:
        # afficher tout le contenu du fichier zip
        zip1.printdir()
        file = file.split('/content/')[1]
        file = file.split('.zip')[0]
        rep = 'Data/' + file
        # On supprime le dossier
        shutil.rmtree(rep, True)
        # On créé les dossier de données
        create_repertory(rep)
        # extraire tous les fichiers
        print('extraction...')
        zip1.extractall(rep)
        print('Terminé!')
# Lister les dossiers
# On liste les dossiers
train_img = []
test_img = []
train_masks = []
test_masks = []
car_models_json = []

# On liste le train
files = glob.glob('Data/train_images' + "/*.jpg")
for file in files:
    train_img.append(file)

files = glob.glob('Data/train_masks' + "/*.jpg")
for file in files:
    train_masks.append(file)
# On liste le test
files = glob.glob('Data/test_images' + "/*.jpg")
for file in files:
    test_img.append(file)

files = glob.glob('Data/test_masks' + "/*.jpg")
for file in files:
    test_masks.append(file)

# On liste les json
files = glob.glob('Data/car_models_json' + "/*.json")
for file in files:
    car_models_json.append(file)

# Train_img
train_img_df = pd.DataFrame(train_img, columns=['train_img'])
train_img_df['ImageId'] = train_img_df['train_img'].str.split('Data/train_images/', expand=True)[1].str.split('.jpg', expand=True)[0]

# Train_masks
train_masks_df = pd.DataFrame(train_masks, columns=['train_masks'])
train_masks_df['ImageId'] = train_masks_df['train_masks'].str.split('Data/train_masks/', expand=True)[1].str.split('.jpg', expand=True)[0]
# Test_img
test_img_df = pd.DataFrame(test_img, columns=['test_img'])
test_img_df['ImageId'] = test_img_df['test_img'].str.split('Data/test_images/', expand=True)[1].str.split('.jpg', expand=True)[0]

# Test_masks
test_masks_df = pd.DataFrame(test_masks, columns=['test_masks'])
test_masks_df['ImageId'] = test_masks_df['test_masks'].str.split('Data/test_masks/', expand=True)[1].str.split('.jpg', expand=True)[0]

# json
car_models_json_df = pd.DataFrame(car_models_json, columns=['car_models_json'])
car_models_json_df['ImageId'] = car_models_json_df['car_models_json'].str.split('Data/car_models_json/', expand=True)[1].str.split('.json', expand=True)[0]

# Import train.csv
train_csv = pd.read_csv("/content/Data/train.csv/train.csv")

# Concat train
train = pd.merge(train_csv, train_img_df, on='ImageId', how='outer')
train = pd.merge(train, train_masks_df, on='ImageId', how='outer')
train.head()

# Création du DataFrame train
images = []
model_type = []
yaw = []
pitche = []
roll = []
x = []
y = []
z = []

for i in list(range(0, train.shape[0])):
    pred_string = train.PredictionString.iloc[i]
    items = pred_string.split(' ')
    resultats = [items[i::7] for i in range(7)]
    model_types, yaws, pitches, rolls, xs, ys, zs = resultats
    model_type.append(model_types)
    yaw.append(yaws)
    pitche.append(pitches)
    roll.append(rolls)
    x.append(xs)
    y.append(ys)
    z.append(zs)
    images.append(train.loc[i, 'ImageId'])
liste = pd.DataFrame([images, model_type, yaw, pitche, roll, x, y, z],
                     index=['ImageId',
                            'model_type',
                            'yaw',
                            'pitche',
                            'roll',
                            'x',
                            'y',
                            'z']).T
liste['nb_car'] = [len(liste['model_type'][i]) for i in range(liste.shape[0])]
liste.head()

resultats = prepare_data(alpha=0.2, nb_pics=0.2, size=512, border=0)
liste_index_train, liste_index_test, liste_index_val = resultats
