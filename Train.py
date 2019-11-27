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
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import math
from keras import models
from keras.layers import Input
from keras.layers import BatchNormalization








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

def score(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.sum(K.abs(y_true/y_true)) / K.sum(K.abs(y_pred/y_true))

def sum_absolute_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.sum(K.abs(y_pred - y_true), axis=-1)
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
    liste_index_train1 = []
    nb_pic = int(len(list(train['train_img'].index))*nb_pics)
    liste_index_train1 = random.sample(list(train['train_img'].index), nb_pic)
    a = int(len(liste_index_train1)*alpha)
    liste_index_test1 = random.sample(liste_index_train1, a)
    for j in liste_index_test1:
        del liste_index_train1[liste_index_train.index1(j)]
    liste_index_val1 = random.sample(liste_index_train1, a)
    for j in liste_index_val1:
        del liste_index_train1[liste_index_val1.index(j)]
    # création du jeu d'entrainement
    for k1 in liste_index_train1:
        img = image_format(k1, border, size)
        img = img.save('Data/Train/' + train['ImageId'][k1] + '.jpg')
    # création du jeu de test
    for k1 in liste_index_test1:
        img = image_format(k1, border, size)
        img = img.save('Data/Test/' + train['ImageId'][k1] + '.jpg')
    # création du jeu de val
    for k1 in liste_index_val1:
        img = image_format(k1, border, size)
        img = img.save('Data/Val/' + train['ImageId'][k1] + '.jpg')
    return liste_index_train1, liste_index_test1, liste_index_val1


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

# Modelisation nb_car
x_train = train['train_img'][liste_index_train]
x_test = train['train_img'][liste_index_test]
x_val = train['train_img'][liste_index_val]
y_train = liste['nb_car'][liste_index_train]/44
y_test = liste['nb_car'][liste_index_test]/44
y_val = liste['nb_car'][liste_index_val]/44

# Pour l'entrainnement
new_train = pd.DataFrame({"x": x_train})
new_train = new_train.join(y_train)

new_test = pd.DataFrame({"x": x_test})
new_test = new_test.join(y_test)

# Pour la validation
new_val = pd.DataFrame({"x": x_val})
new_val = new_val.join(y_val)
# On corrige les chemins d'accès
new_train['x'] = new_train['x'].str.replace('train_images', 'Train')
new_test['x'] = new_test['x'].str.replace('train_images', 'Test')
new_val['x'] = new_val['x'].str.replace('train_images', 'Val')

# parametres
img_size = 512
nb_conv = 6
nb_dense = 4
units = 512
dropout = 0.5
# construction du modèle
input_shape = (img_size, img_size, 3)

x = Input(shape=input_shape)
l = x
l = Convolution2D(filters=2, kernel_size=[1, 1],
                  strides=1, activation="relu")(l)
l = BatchNormalization()(l)
l = MaxPooling2D()(l)

for i in list(range(nb_conv)): 
    l = Convolution2D(filters=2*(2**(i//2+1)), kernel_size=[3, 3], strides=1,
                      activation="relu")(l)
    l = BatchNormalization()(l)
    l = MaxPooling2D(pool_size=(2, 2), strides=2)(l)
# Couche Flattening
l = Flatten()(l)


l = Dense(units=1024, activation="relu")(l)
l = Dropout(dropout)(l)



l = Dense(units=1, activation="linear")(l)

first_model = models.Model(x, l)
first_model.compile(loss=sum_absolute_error, optimizer='adam', metrics=[score])


first_model.summary()



train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_dataframe(new_train,
                                                 directory=None,
                                                 x_col='x',
                                                 y_col='nb_car',
                                                 weight_col=None,
                                                 target_size=(img_size,
                                                              img_size),
                                                 color_mode='rgb',
                                                 classes=None,
                                                 class_mode='raw',
                                                 batch_size=32,
                                                 shuffle=True,
                                                 seed=None,
                                                 save_to_dir=None,
                                                 save_prefix='',
                                                 save_format='jpg',
                                                 subset=None,
                                                 interpolation='nearest', 
                                                 validate_filenames=True)

test_set = test_datagen.flow_from_dataframe(new_test,
                                            directory=None,
                                            x_col='x',
                                            y_col='nb_car',
                                            weight_col=None,
                                            target_size=(img_size,
                                                         img_size),
                                            color_mode='rgb',
                                            classes=None,
                                            class_mode='raw',
                                            batch_size=32,
                                            shuffle=True,
                                            seed=None,
                                            save_to_dir=None,
                                            save_prefix='',
                                            save_format='jpg',
                                            subset=None,
                                            interpolation='nearest', 
                                            validate_filenames=True)

val_set = test_datagen.flow_from_dataframe(new_val,
                                            directory=None,
                                            x_col='x',
                                            y_col='nb_car',
                                            weight_col=None,
                                            target_size=(img_size,
                                                         img_size),
                                            color_mode='rgb',
                                            classes=None,
                                            class_mode='raw',
                                            batch_size=32,
                                            shuffle=True,
                                            seed=None,
                                            save_to_dir=None,
                                            save_prefix='',
                                            save_format='jpg',
                                            subset=None,
                                            interpolation='nearest', 
                                            validate_filenames=True)

checkpoint = ModelCheckpoint("nb_car.h5", monitor='val_loss',
                             verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto', period=1)



History = first_model.fit_generator(training_set,
                                    steps_per_epoch=int(math.ceil(len(training_set.filenames)/32)),
                                    epochs=100,
                                    validation_data=test_set,
                                    validation_steps=int(math.ceil(len(test_set.filenames)/32)),
                                    callbacks = [checkpoint],
                                    workers=2)


plt.figure(figsize=(30,10))
y_hat = []
for i in list(range(len(val_set.filepaths))):
    test_image = Image.open(val_set.filepaths[i])
    test_image_full = np.array(test_image)
    test_image = test_image.resize((img_size, img_size), resample=0)
    test_image = np.array(test_image)/255
    test_image = np.expand_dims(test_image, axis=0)
    result = first_model.predict(test_image)
    y_hat.append(int(result[0]*44))

    
plt.plot(y_hat)
plt.plot(list(liste['nb_car'][liste_index_val]))
ecart = np.mean(np.abs(np.array(y_hat) - liste['nb_car'][liste_index_val]))
print('mean absolute error: ', ecart)

# Modélisation (position)
x_train = train['train_img'][liste_index_train]
x_test = train['train_img'][liste_index_test]
x_val = train['train_img'][liste_index_val]
y_train = train['PredictionString'].str.split(' ', expand=True).astype('float32').loc[liste_index_train,:].fillna(value=0)

#normalize_min = list(train['PredictionString'].str.split(' ', expand=True).astype('float32').loc[liste_index_train,:].fillna(value=0).min())
normalize_min = 0
normalize_max = list(abs(train['PredictionString'].str.split(' ', expand=True).astype('float32').loc[liste_index_train,:].fillna(value=0)).max())

y_train = (y_train-normalize_min)/normalize_max
y_train = y_train.fillna(value=0)
y_test = train['PredictionString'].str.split(' ', expand=True).astype('float32').loc[liste_index_test,:].fillna(value=0)
y_test = (y_test-normalize_min)/normalize_max
y_test = y_test.fillna(value=0)
y_val = train['PredictionString'].str.split(' ',expand=True).astype('float32').loc[liste_index_val,:].fillna(value=0)
y_val = (y_val-normalize_min)/normalize_max
y_val = y_val.fillna(value=0)

# Pour l'entrainnement
new_train = pd.DataFrame({"x": x_train})
new_train = new_train.join(y_train)

new_test = pd.DataFrame({"x": x_test})
new_test = new_test.join(y_test)

# Pour la validation
new_val = pd.DataFrame({"x": x_val})
new_val = new_val.join(y_val)

# On corrige les chemins d'accès
new_train['x'] = new_train['x'].str.replace('train_images', 'Train')
new_test['x'] = new_test['x'].str.replace('train_images', 'Test')
new_val['x'] = new_val['x'].str.replace('train_images', 'Val')

# parametres
col = list(range(131, 141))
img_size = 512
nb_conv = 6
nb_dense = 4
units = 512
dropout = 0.5
# construction du modèle
input_shape = (img_size, img_size, 3)
for k in col:
    x = Input(shape=input_shape)
    l = x
    l = Convolution2D(filters=2, kernel_size=[1, 1], strides=1, activation="relu")(l)
    l = BatchNormalization()(l)
    l = MaxPooling2D()(l)

    for i in list(range(nb_conv)): 
        l = Convolution2D(filters=2*(2**(i//2+1)), kernel_size=[3, 3], strides=1, activation="relu")(l)
        l = BatchNormalization()(l)
        l = MaxPooling2D(pool_size=(2, 2), strides=2)(l)
    # Couche Flattening
    l = Flatten()(l)


    l = Dense(units=1024, activation="relu")(l)
    l = Dropout(dropout)(l)



    l = Dense(units=1, activation="linear")(l)

    train_model = models.Model(x, l)
    train_model.compile(loss=sum_absolute_error, optimizer='adam', metrics=[score])


    train_model.summary()



    train_datagen = ImageDataGenerator(rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_dataframe(new_train,
                                                    directory=None,
                                                    x_col='x',
                                                    y_col=k,
                                                    weight_col=None,
                                                    target_size=(img_size, img_size),
                                                    color_mode='rgb',
                                                    classes=None,
                                                    class_mode='raw',
                                                    batch_size=32,
                                                    shuffle=True,
                                                    seed=None,
                                                    save_to_dir=None,
                                                    save_prefix='',
                                                    save_format='jpg',
                                                    subset=None,
                                                    interpolation='nearest', 
                                                    validate_filenames=True)

    test_set = test_datagen.flow_from_dataframe(new_test,
                                                directory=None,
                                                x_col='x',
                                                y_col=k,
                                                weight_col=None,
                                                target_size=(img_size, img_size),
                                                color_mode='rgb',
                                                classes=None,
                                                class_mode='raw',
                                                batch_size=32,
                                                shuffle=True,
                                                seed=None,
                                                save_to_dir=None,
                                                save_prefix='',
                                                save_format='jpg',
                                                subset=None,
                                                interpolation='nearest', 
                                                validate_filenames=True)

    val_set = test_datagen.flow_from_dataframe(new_val,
                                                directory=None,
                                                x_col='x',
                                                y_col=k,
                                                weight_col=None,
                                                target_size=(img_size, img_size),
                                                color_mode='rgb',
                                                classes=None,
                                                class_mode='raw',
                                                batch_size=32,
                                                shuffle=True,
                                                seed=None,
                                                save_to_dir=None,
                                                save_prefix='',
                                                save_format='jpg',
                                                subset=None,
                                                interpolation='nearest', 
                                                validate_filenames=True)
    model_name = 'model' + str(k) + '.h5'
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, verbose=1, mode='auto')


    History = train_model.fit_generator(training_set,
                                        steps_per_epoch=int(math.ceil(len(training_set.filenames)/32)), # len du jeux d'entrainement / batch
                                        epochs=50,
                                        validation_data=test_set,
                                        validation_steps=int(math.ceil(len(test_set.filenames)/32)), # len du jeux de test / batch
                                        callbacks = [checkpoint, early],
                                        workers=2)


    plt.figure(figsize=(30, 10))
    y_hat = []
    for i in list(range(len(val_set.filepaths))):
        test_image = Image.open(val_set.filepaths[i])
        test_image_full = np.array(test_image)
        test_image = test_image.resize((img_size, img_size), resample=0)
        test_image = np.array(test_image)/255
        test_image = np.expand_dims(test_image, axis=0)
        result = train_model.predict(test_image)
        y_hat.append(result[0])
      
    plt.plot(y_hat)
    plt.plot(list(new_val[k]))
    plt.show()
    ecart = np.mean(np.abs(np.array(y_hat) - list(new_val[k])))
    print('mean absolute error: ', ecart)