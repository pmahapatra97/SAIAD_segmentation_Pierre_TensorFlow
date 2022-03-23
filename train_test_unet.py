# -*- coding: utf-8 -*-

import sys
import os

import shutil
import errno

import numpy as np
import pandas as pd 
from time import time
from datetime import datetime
import tensorflow as tf

import pickle

# from tensorflow.keras import losses, metrics, layers
import tensorflow.keras.backend as K
# from tensorflow.keras.utils import to_categorical

from PIL import Image
import collections

import matplotlib.pyplot as plt

# from unet import unet 

# from unet.trainer import Trainer

# import unet.metrics as me

from typing import Tuple

import constant
import data
# from postTreatments import postProcessing

from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Conv2D, Conv3D
from tensorflow.keras.models import Model

import segmentation_models as sm

import argparse


# set randomness
np.random.seed(1)
tf.random.set_seed(2)

def get_from_parser(verbose=True):
    # create parser 
    parser = argparse.ArgumentParser(description="Train test script Unet")

    # add arguments
    parser.add_argument("patients", nargs='+')
    parser.add_argument("--structure", default="tumeur",help="structure to segment: tumeur reinPatho")
    parser.add_argument("--ovassion",  action="store_true", help="use ovassion")
    parser.add_argument("--ovassion_r",  action="store_true", help="use ovassion_r")
    parser.add_argument("-l","--learning_rate", type=float, default=1e-4, help="learning rate for training")
    parser.add_argument("--gap",type=int, default=None,help="gap for ovassion training")
    parser.add_argument("--context",type=int, default=0,help="number of neighbour input scans")
    parser.add_argument("-n","--name", default="unet_default",help="experiment name")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--ovassion_exp", help="experiment of the model used for ovassion training")
    parser.add_argument("--encoder_freeze", type=float, default=0.0, help="freezing rate of the encoder")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--selection", default="gap", help="selection type of ovassion slices")
    parser.add_argument("--nbSelected",type=int, default=None,help="number of selected slices for ovassion training")
    parser.add_argument("--shift",type=int, default=None,help="shift in selection")
    parser.add_argument("--augmentation",  action="store_true", help="use ovassion_r")
    parser.add_argument("--enhanced",  action="store_true", help="use ovassion_r")
    parser.add_argument("--variable",  help="name of the variable to vary")
    parser.add_argument("--variable_list", nargs='*', help="values of the variable to varry")


    # get input arguments
    args = parser.parse_args()

    if args.structure == "tumeur_reinPatho":
        # args.loss = sm.losses.categorical_focal_dice_loss
        args.loss = sm.losses.categorical_crossentropy
        args.loss = sm.losses.CategoricalFocalLoss()+sm.losses.DiceLoss(class_indexes=[1,2])
    else:
        args.loss = sm.losses.binary_focal_dice_loss
    
    args.experiment = args.name
    args.mode="normal"
    if args.ovassion:
        args.mode="ovassion"
    if args.ovassion_r:
        args.mode="ovassion_r"

    if args.ovassion_exp is None:
        args.ovassion_exp = args.experiment
    if args.mode=="ovassion":
        if args.selection=="gap":
            args.experiment+="_{}_ovassion".format(args.gap)
        elif args.selection=="number":
            args.experiment+="_{}s_ovassion".format(args.nbSelected)
    elif args.mode=="ovassion_r":
        args.experiment+="_{}_ovassion_r".format(args.gap)

    if verbose:
        print("Parameters:")

        print("Patients: ",args.patients)
        print("Structure: ",args.structure)
        print("Experiment: ",args.experiment)
        print("Context: ", args.context)

        print("Encoder freezing rate: ", args.encoder_freeze)
        print("Dropout rate: ", args.dropout_rate)
        print("Learning rate: ", args.learning_rate)
        print("Nb epochs: ",args.epochs)
        print("Batch size: ",args.batch_size)
        print("Loss: ",args.loss.name)
        print("Augmentation: ",str(args.augmentation))
        print("Enhanced: ", str(args.enhanced))

        print("Mode: ", args.mode)
        if args.mode != "normal":
            print("Ovassion experiment: ", args.ovassion_exp)
            print("Ovassion selection: ", args.selection)
            print("Ovassion gap: ", args.gap)
            print("Ovassion nbSelected: ", args.nbSelected)
    return args

def store_experiment(args):

    add_folder(constant.RESULTS_DIR_EXP.format(args.experiment))
    add_folder(os.path.join(constant.RESULTS_DIR_EXP.format(args.experiment),"logs"))
    expPath = constant.RESULTS_DIR_EXP.format(args.experiment)+'description.txt'
    expFile = open(expPath,'w')

    expFile.write("Parameters:"+"\n\n")

    expFile.write("Patients: "+str(args.patients)+"\n")
    expFile.write("Structure: "+args.structure+"\n")
    expFile.write("Experiment: "+args.experiment+"\n")
    expFile.write("Context: "+str(args.context)+"\n\n")

    expFile.write("Encoder freezing rate: "+str(args.encoder_freeze)+"\n")
    expFile.write("Dropout rate: "+str(args.dropout_rate)+"\n")
    expFile.write("Learning rate: "+str(args.learning_rate)+"\n")
    expFile.write("Nb epochs: "+str(args.epochs)+"\n")
    expFile.write("Batch size: "+str(args.batch_size)+"\n")
    expFile.write("Loss: "+args.loss.name+"\n")
    expFile.write("Augmentation: "+str(args.augmentation)+"\n")
    expFile.write("Enhanced: "+str(args.enhanced)+"\n\n")

    expFile.write("Mode: "+args.mode+"\n")
    expFile.write("Ovassion experiment: "+args.ovassion_exp+"\n")
    expFile.write("Ovassion selection: "+args.selection+"\n")
    expFile.write("Ovassion gap: "+str(args.gap)+"\n")
    expFile.write("Ovassion gap: "+str(args.nbSelected)+"\n\n")

    expFile.close()

def add_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    

def create_result_folders(results_path):
    # Delete results directory containing previous results.
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    add_folder(results_path)

    add_folder(results_path + 'classes')
    add_folder(results_path + 'std')

    add_folder(results_path + 'logs')
      
    add_folder(results_path + 'CRF')          
    
    add_folder(results_path + 'CRF/color')    
    add_folder(results_path + 'CRF/Map')   

def save_results(out, idx, path):
    res = np.array(out, dtype=np.uint8)
    
    img = Image.fromarray(res)
    img.save(path+'classes/{}.png'.format(idx))
    colorized = np.zeros((len(res), len(res[0]), 3), np.uint8)
    for i in range(len(res)):
        for j in range(len(res[0])):
            if res[i][j] == 1:
                colorized[i][j][0] = 72
                colorized[i][j][1] = 119
                colorized[i][j][2] = 71
            if res[i][j] == 2:
                colorized[i][j][0] = 90
                colorized[i][j][1] = 52
                colorized[i][j][2] = 41

    colorized_ = Image.fromarray(colorized, 'RGB')
    colorized_.save(path+'{}.png'.format(idx))

def save_std(out, idx, path):
    res = np.array(out, dtype=np.float32)
    np.save(path+'std/{}.npy'.format(idx),res)



def save_training_plot(history, training_plot_path):
    pd.DataFrame(history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    
    plt.savefig(training_plot_path+".png")

    df = pd.DataFrame.from_dict(history)
    df.to_csv(training_plot_path+".csv")
    df.to_dict(orient="list")

def crop_labels_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label, weight=None):
        if weight is None:
            return image, crop_to_shape(label, shape)
        else:
            return image, crop_to_shape(label, shape), weight
    return crop

def crop_to_shape(data, shape: Tuple[int, int, int]):
    """
    Crops the array to the given image shape by removing the border

    :param data: the array to crop, expects a tensor of shape [batches, nx, ny, channels]
    :param shape: the target shape [batches, nx, ny, channels]
    """
    diff_nx = (data.shape[0] - shape[0])
    diff_ny = (data.shape[1] - shape[1])

    if diff_nx == 0 and diff_ny == 0:
        return data

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[0] == shape[0]
    assert cropped.shape[1] == shape[1]
    return cropped



"""
Calcul du DICE global entre deux ensembles de segmentations 
(Ne prend pas en compte le Dice de chaque segmentation mais calcul un Dice global)
Args :
    listSeg (collections.OrderedDict()): Ensemble de segmentations calculées
    listSegVT (collections.OrderedDict()): Ensemble des segmentations VT
    n_classes(int) : nombre de classes dans les segmentations
Returns : 
    float : Valeur du Dice 
"""
def dice_hm(listSeg, listSegVT, n_classes):
    dice = np.zeros(n_classes)
    for i in range(n_classes):
        vp = 0
        fp = 0
        fn = 0
        for nbImg,y_pred in listSeg.items():
            y_true = listSegVT[nbImg]
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            vp += np.sum(np.logical_and(y_pred == i, y_true == i))
            fp += np.sum(np.logical_and(y_pred == i, y_true != i))
            fn += np.sum(np.logical_and(y_pred != i, y_true == i))
        if vp==0 and fp ==0 and fn == 0 :
            dice[i]=1
        else :
            dice[i] = (2*vp)/(2*vp +fp+fn)
    return dice


"""
Calcul de l'IoU entre deux ensembles de segmentations 
(Ne prend pas en compte l'IoU de chaque segmentation mais calcul un IoU global)
Args :
    listSeg (collections.OrderedDict()): Ensemble de segmentations calculées
    listSegVT (collections.OrderedDict()): Ensemble des segmentations VT
    n_classes(int) : nombre de classes dans les segmentations
Returns : 
    float : Valeur de l'IoU
"""
def iu_hm(listSeg, listSegVT, n_classes):
    iu = np.zeros(n_classes)
    for i in range(n_classes):
        vp = 0
        fp = 0
        fn = 0
        for nbImg,y_pred in listSeg.items():
            y_true = listSegVT[nbImg]
            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            vp += np.sum(np.logical_and(y_pred == i, y_true == i))
            fp += np.sum(np.logical_and(y_pred == i, y_true != i))
            fn += np.sum(np.logical_and(y_pred != i, y_true == i))
        if vp==0 and fp ==0 and fn == 0 :
            iu[i]=1
        else :
            iu[i] = vp/(vp+fp+fn)
    return iu

def change_model(model, new_input_shape=(None, 40, 40, 3)):
    # replace input shape of first layer
    model._layers[1].batch_input_shape = new_input_shape

    # feel free to modify additional parameters of other layers, for example...
    model._layers[2].pool_size = (8, 8)
    model._layers[2].strides = (8, 8)

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())
    new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
    X = np.random.rand(10, 40, 40, 3)
    y_pred = new_model.predict(X)
    print(y_pred)

    return new_model

def old_init_unet(path_old_save_model, args):
    
    np.random.seed(1)
    tf.random.set_seed(2)

    base_model = sm.Unet('vgg16', encoder_weights = "imagenet", encoder_freeze=True)

    inp = Input(shape=(None, None, 1))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    unet_model = Model(inp, out, name=base_model.name)
    # sm.utils.set_trainable(unet_model, recompile=False)
    if args.mode=="ovassion" or args.mode =="ovassion_r":
        old_unet_model = tf.keras.models.load_model(path_old_save_model,
           custom_objects={args.loss.name:args.loss,
                        'iou_score':sm.metrics.iou_score,
                        'f1-score':sm.metrics.f1_score
                        })
        unet_model.set_weights(old_unet_model.get_weights())
        print('loaded model: '+ path_old_save_model)
    
    unet_model.compile(
        'Adam',
        loss=args.loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
    )
    
    unet_model.summary()
    base_model.summary()

    return unet_model

def init_unet(path_old_save_model, args):
    
    np.random.seed(1)
    tf.random.set_seed(2)

    if args.structure=="tumeur_reinPatho":
        unet_model = sm.myUnet('vgg16', encoder_weights = "imagenet", classes = 3, activation="softmax", 
            encoder_freeze=args.encoder_freeze, dropout_rate=args.dropout_rate)
    else:
        unet_model = sm.myUnet('vgg16', encoder_weights = "imagenet", 
            encoder_freeze=args.encoder_freeze, dropout_rate=args.dropout_rate)

   
    
    if args.mode=="ovassion" or args.mode =="ovassion_r":
        old_unet_model = tf.keras.models.load_model(path_old_save_model,
           custom_objects={args.loss.name:args.loss,
                        'iou_score':sm.metrics.iou_score,
                        'f1-score':sm.metrics.f1_score
                        })
        unet_model.set_weights(old_unet_model.get_weights())
        print('loaded model: '+ path_old_save_model)
    
    unet_model.compile(
        'Adam',
        loss=args.loss,
        metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
    )
    
    unet_model.summary()

    return unet_model

def load_model(path_save_model,args):
    model = tf.keras.models.load_model(path_save_model,
    custom_objects={args.loss.name:args.loss,
                    'iou_score':sm.metrics.iou_score,
                    'f1-score':sm.metrics.f1_score
                    })
    return model


def load_model_to_dropout(path_save_model,args):

    model = sm.myUnet('vgg16', encoder_weights = "imagenet", encoder_freeze=True)
    old_model = tf.keras.models.load_model(path_save_model,
        custom_objects={args.loss.name:args.loss,
                        'iou_score':sm.metrics.iou_score,
                        'f1-score':sm.metrics.f1_score
                        })
    model.set_weights(old_model.get_weights())
    return model


def train(model, data_train_test, args):

    print("TRAINING")
    #ENTRAINEMENT
    print("BATCH_SIZE=",args.batch_size)
    print("LEARNING_RATE=",args.learning_rate)


    np.random.seed(1)
    tf.random.set_seed(2)

    if args.mode != "ovassion_r":

        # data_train_arr = data_train_test.get_train()
        # data_test_arr  = data_train_test.get_test()
        # nb_train = data_train_arr[0].shape[0]
        # print(nb_train)
        # data_train = tf.data.Dataset.from_tensor_slices(data_train_arr).shuffle(nb_train,reshuffle_each_iteration=True).prefetch(10)
        # data_test  = tf.data.Dataset.from_tensor_slices(data_test_arr).prefetch(10)
        
        # print([np.sum(a) for a in list(data_train.take(count=5).as_numpy_iterator())])
        # print([np.sum(a) for a in list(data_train.take(count=5).as_numpy_iterator())])

        # prediction_shape = model.predict(data_train.take(count=1).batch(batch_size=1)).shape[1:]
        # data_train = data_train.map(crop_labels_to_shape(prediction_shape)).batch(args.batch_size)
        # data_test = data_test.map(crop_labels_to_shape(prediction_shape)).batch(args.batch_size)
        print(args.batch_size)
        data_train, data_val = data_train_test.get_train_val_gen(args.batch_size, args.augmentation, args.enhanced)


        logdir = os.path.join(constant.RESULTS_DIR_EXP.format(args.experiment),"logs", args.patient+"_"+datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
        
        print(data_train_test.nbSlices)
        print(data_train_test.nbTrain)
        print(int(data_train_test.nbTrain*0.9))
        print(int(data_train_test.nbTrain*0.1))
        history = model.fit(data_train,
                                validation_data=data_val,
                                epochs=args.epochs,
                                steps_per_epoch = int(data_train_test.nbTrain*0.9) // args.batch_size + 1 ,
                                validation_steps = int(data_train_test.nbTrain*0.1) // args.batch_size + 1 ,
                                callbacks=[tensorboard_callback])
        his = history.history

    else:

        nb_step = data_train_test.nbSteps
        # nb_epoch_step = int(args.epochs/nb_step)
        nb_epoch_step = args.epochs
        history_list =[]
        def weight_step(step):
            # return 1
            return np.exp(-step/5)

        for sstep in range(nb_step+1):
            print(data_train_test.step_idx(sstep))
            print(weight_step(sstep))
        weights = [np.full(len(data_train_test.step_idx(sstep)),weight_step(sstep)) for sstep in range(nb_step+1)]
        n_0 = len(data_train_test.step_idx(0))

        for k in range(1,nb_step+1):
            weights[k] = weights[k] * (n_0/(data_train_test.nbSlices-n_0))
        print(weights)

        data_k = [data_train_test.get_step(k) for k in range(nb_step+1)]
        data_scan = [data_k[k][0] for k in range(nb_step+1)]
        data_label = [data_k[k][1] for k in range(nb_step+1)]
        
        for step in range(nb_step):
            

            if step!=0:
                old_data_pred = data_pred[:]
            
            data_pred = [np.expand_dims(data_train_test.get_step(k,"prediction")[1],-1) for k in range(nb_step+1)]
            data_pred[0] = data_label[0]
            
            if step!=0:
                print(sum([old_data_pred[k].sum() for k in range(0,step)]))
                print(sum([data_pred[k].sum() for k in range(0,step)]))
            
            print(len(data_pred))
            print(len(data_label))
            print([dp.shape for dp in data_pred])
            print([dl.shape for dl in data_label])
            data_train_arr=(np.concatenate([data_scan[k] for k in range(0,step+1)]),
                            np.concatenate([data_pred[k] for k in range(0,step+1)]),
                            np.concatenate([weights[k] for k in range(0,step+1)]))
            print(np.concatenate([weights[k] for k in range(0,step+1)]))
            data_test_arr =(np.concatenate([data_scan[k] for k in range(step+1,nb_step+1)]),
                            np.concatenate([data_label[k] for k in range(step+1,nb_step+1)]))
            
            nb_train = data_train_arr[0].shape[0]
            data_train = tf.data.Dataset.from_tensor_slices(data_train_arr).shuffle(nb_train,reshuffle_each_iteration=True).prefetch(10)
            data_test  = tf.data.Dataset.from_tensor_slices(data_test_arr).prefetch(10)

            prediction_shape = model.predict(data_train.take(count=1).batch(batch_size=1)).shape[1:]
            data_train = data_train.map(crop_labels_to_shape(prediction_shape)).batch(args.batch_size)
            data_test = data_test.map(crop_labels_to_shape(prediction_shape)).batch(args.batch_size)
            history = model.fit(data_train,
                                validation_data=data_test,
                                epochs=nb_epoch_step)
            history_list.append(history.history)
            
            for sstep in range(1,step+2):
                pred_images = data_train_test.get_step(sstep)[0]
                pred = model.predict(pred_images)
                pred_mask = K.greater(pred,0.5)
                pred_final = K.cast(np.squeeze(pred_mask), np.uint8)
                data_train_test.set_pred(sstep, pred_final)

        # nb_step = len(data_train_test)-1
        # nb_epoch_step = int(args.epochs/nb_step)
        # # nb_epoch_step = NB_EPOCHS
        # history_list =[]
        # data_pred = [[] for k in range(len(data_train_test))]
        # data_pred[0] = data_train_test[0][1]
        # print([(dtidx[0].shape,dtidx[1].shape)  for dtidx in data_train_test])
        # weights = [np.full(data_train_test[k][0].shape[0], np.exp(-k/5)) for k in range(nb_step+1)]
        # print(weights)
        # for step in range(nb_step):
            
        #     data_train_arr=(np.concatenate([data_train_test[k][0] for k in range(0,step+1)]),
        #                     np.concatenate([data_pred[k] for k in range(0,step+1)]),
        #                     np.concatenate([weights[k] for k in range(0,step+1)]))
        #     print(np.concatenate([weights[k] for k in range(0,step+1)]))
        #     data_test_arr =(np.concatenate([data_train_test[k][0] for k in range(step+1,nb_step+1)]),
        #                     np.concatenate([data_train_test[k][1] for k in range(step+1,nb_step+1)]))
        #     nb_train = data_train_arr[0].shape[0]
        #     data_train = tf.data.Dataset.from_tensor_slices(data_train_arr).shuffle(nb_train,reshuffle_each_iteration=True).prefetch(10)
        #     data_test  = tf.data.Dataset.from_tensor_slices(data_test_arr).prefetch(10)

        #     prediction_shape = model.predict(data_train.take(count=1).batch(batch_size=1)).shape[1:]
        #     data_train = data_train.map(crop_labels_to_shape(prediction_shape)).batch(args.batch_size)
        #     data_test = data_test.map(crop_labels_to_shape(prediction_shape)).batch(args.batch_size)
        #     history = model.fit(data_train,
        #                         validation_data=data_test,
        #                         epochs=nb_epoch_step)
        #     history_list.append(history.history)
            
        #     for sstep in range(1,step+2):
        #         pred_images = data_train_test[sstep][0]
        #         print(pred_images.shape)
        #         # for image, label in :

        #         # image = image[tf.newaxis,...]
            
        #         pred = model.predict(pred_images)
        #         pred_mask = K.greater(pred,0.5)
        #         data_pred[sstep] = pred_mask
        #         # pred_final = K.cast(np.squeeze(pred_mask), np.uint8)


        his = {}
        for k in history_list[0].keys():
            his[k] = list(np.concatenate(list(hist[k] for hist in history_list)))
    
    return model, his

# def test_predict(model, data_train_test, data_idx, patient_test, path_res):
def test_predict(model, data_train_test, path_res):

    # l_train_test = len(data_train_test)
    # data_test_arr =(np.concatenate([data_train_test[k][0] for k in range(l_train_test)]),
    #                 np.concatenate([data_train_test[k][1] for k in range(l_train_test)]),
    #                 np.concatenate([[idd[0] for idd in data_idx[k]] for k in range(l_train_test)]),
    #                 np.concatenate([[idd[1] for idd in data_idx[k]] for k in range(l_train_test)]))
    # print(len(data_test_arr))
    # print([b.shape for b in data_test_arr])
    # data_test  = tf.data.Dataset.from_tensor_slices(data_test_arr).prefetch(10)

    create_result_folders(path_res)
    print("RESULTS PATH: ",path_res)

    #data_test, = data_train_test.get_test_gen()
    for idx, s in data_train_test.patients[data_train_test.patient_test].slices.items():
 #       t1 = time()
#        print(idx)
        image = s.scan[tf.newaxis,...]
        pred = model.predict(image)[0]

        if pred.shape[2]>1:
            pred_mask = np.argmax(pred, axis = 2)
            pred_final = K.cast(pred_mask, np.uint8)
            s.set_pred(tf.keras.utils.to_categorical(pred_final,num_classes=3))
        else:
            pred_mask = K.greater(pred,0.5)
            pred_final = K.cast(np.squeeze(pred_mask), np.uint8)
            s.set_pred(pred_final[..., np.newaxis])

        save_results(pred_final, idx, path_res)


 # for image, label, patient, idx in data_test.take(len(list(data_test))):
    #     if patient==patient_test:
    #         image = image[tf.newaxis,...]
    #         #Lancer la prediction sur les images test
            
    #         pred = model.predict(image)
    #         pred_mask = K.greater(pred,0.5)
    #         pred_final = K.cast(np.squeeze(pred_mask), np.uint8)

    #         save_results(pred_final, idx, path_res)
            
            # if feature_for_crf :
            #     imgSoftmax = np.exp(np.concatenate((1-pred[0],pred[0]),2), dtype=np.float128)
            #     imgSoftmax = imgSoftmax.transpose((2,0,1))
            #     imgSoftmax = imgSoftmax /(imgSoftmax.sum(axis=0))
            #     img1 = imgSoftmax[0].astype(np.float64)
            #     img2 = imgSoftmax[1].astype(np.float64)
            #     save_results_crf(img1, img2, idx, path_res+'CRF/Map/')

def test_predict_dropout(model, data_train_test, path_res):

    create_result_folders(path_res)
    print("RESULTS PATH: ",path_res)
    t1 = time()
    for idx, s in data_train_test.patients[data_train_test.patient_test].slices.items():
        image = s.scan[tf.newaxis,...]
        # print(idx)
        pred_dropout = []

        #pred = model.predict(image)
        # t2=time()
        for i in range(10):
            #pred_dropout.append(model(image, training=False)[0])   # runs the model in training mode
            pred_dropout.append(model.predict(image)[0])   # runs the model in training mode
            # print(time()-t2)
        # print(time()-t1)
        pred_dropout = np.array(pred_dropout)
        # print(pred_dropout.shape)
        pred_mean = pred_dropout.mean(axis=0)
        # print(pred_mean.shape)
        pred_std = pred_dropout.std(axis=0)
        save_std(pred_std, idx, path_res)

        if pred_mean.shape[2]>1:
            pred_mask = np.argmax(pred_mean, axis = 2)
            pred_final = K.cast(pred_mask, np.uint8)
            s.set_pred(tf.keras.utils.to_categorical(pred_final,num_classes=3))
        else:
            pred_mask = K.greater(pred_mean,0.5)
            pred_final = K.cast(np.squeeze(pred_mask), np.uint8)
            s.set_pred(pred_final[..., np.newaxis])

        # pred_mask = np.squeeze(K.greater(pred_mean,0.5))
        # pred_final = K.cast(pred_mask, np.uint8)
        # # print(pred_final.shape)

        s.set_std(pred_std)
        # print(time()-t1)
        # print(pred_final.shape)
        save_results(pred_final, idx, path_res)
    print(time()-t1)



def test_similarity(model, data_train_test, path_res):
    encoder_model = Model(model.input, model.layers[25].output) 
    scans = data_train_test.get_patient_test()[0]
    encoder_features=[]
    for idx, s in data_train_test.patients[data_train_test.patient_test].slices.items():
        encoder_features.append(encoder_model.predict(s.scan[tf.newaxis,...])[0])

    flatten_features = np.array(encoder_features).reshape((len(encoder_features),-1))
    
    sim_mat = cosine_similarity(flatten_features)

    data_train_test.patients[data_train_test.patient_test].set_similarity(sim_mat)
    np.save(path_res+'sim.npy',sim_mat)


        
def get_predict(data_train_test, path_res):

    # l_train_test = len(data_train_test)
    # data_test_arr =(np.concatenate([data_train_test[k][0] for k in range(l_train_test)]),
    #                 np.concatenate([data_train_test[k][1] for k in range(l_train_test)]),
    #                 np.concatenate([[idd[0] for idd in data_idx[k]] for k in range(l_train_test)]),
    #                 np.concatenate([[idd[1] for idd in data_idx[k]] for k in range(l_train_test)]))
    # print(len(data_test_arr))
    # print([b.shape for b in data_test_arr])
    # data_test  = tf.data.Dataset.from_tensor_slices(data_test_arr).prefetch(10)

    print("RESULTS PATH: ",path_res)

    if data_train_test.structure == "tumeur_reinPatho":
        for idx, s in data_train_test.patients[data_train_test.patient_test].slices.items():
            pred = Image.open(path_res+'/classes/{}.png'.format(idx)).convert('L')
            s.set_pred(tf.keras.utils.to_categorical(pred,num_classes=3))
    else:
        for idx, s in data_train_test.patients[data_train_test.patient_test].slices.items():
            pred = Image.open(path_res+'/classes/{}.png'.format(idx)).convert('L')
            s.set_pred(np.array(pred, dtype=np.uint8)[..., np.newaxis])


# def test_metrics(data_idx, experiment, patient, mode):
def test_metrics(data_train_test, path_res, logPath= '/logs/resultDiceIU.txt'):

    patient_test = data_train_test.patients[data_train_test.patient_test]
    patient_test.compute_metrics()
    dice = patient_test.dice
    iu = patient_test.iou
    dice_patient = patient_test.dice_patient
    iu_patient = patient_test.iou_patient
    logPathDL = path_res+ logPath
    logFile = open(logPathDL,'w')
    if data_train_test.structure == "tumeur_reinPatho":
        structures = ["tumeur", "reinPatho", "tumeur_reinPatho"]
        metric_results_patient = [{"Patient":data_train_test.patient_test, "structure":structures[k], "Dice":dice[k], "IoU": iu[k] ,"DicePatient":dice_patient[k], "IoUPatient": iu_patient[k]} for k in range(len(structures))]
        for k in range(3):
            logFile.write("DICE {}: ".format(structures[k])+str(dice[k])+"\n")
            logFile.write("IU {}: ".format(structures[k])+str(iu[k])+"\n")
            logFile.write("DICE_Patient {}: ".format(structures[k])+str(dice_patient[k])+"\n")
            logFile.write("IU_Patient {}: ".format(structures[k])+str(iu_patient[k])+"\n")
            print("Deep Learning, calcul Dice Iu : ")
            print('DICE_mean {}: '.format(structures[k])+str(dice[k]*100))
            print('IU_MEAN {}: '.format(structures[k])+str(iu[k]*100))
            print('DICE_Patient {}: '.format(structures[k])+str(dice_patient[k]*100))
            print('IU_Patient {}: '.format(structures[k])+str(iu_patient[k]*100))
        logFile.close()
            

        metric_results_slices = [{"Patient":s.patient_name, "idx":s.idx, "structure":structures[k],
                    "nbPixels":s.nbPixels[k], "tumor_size":s.positive[k],
                    "tp":s.true_positive[k], "fp":s.false_positive[k], "fn":s.false_negative[k],
                    "precision":s.precision[k], "recall":s.recall[k], "dice":s.dice[k], "iou": s.iou[k]} for k in range(len(structures)) for s in data_train_test.patients[data_train_test.patient_test].slices.values()]
    else:
        metric_results_patient = [{"Patient":data_train_test.patient_test, "structure":data_train_test.structure, "Dice":dice[0], "IoU": iu[0] ,"DicePatient":dice_patient[0], "IoUPatient": iu_patient[0]}]
        logFile.write("DICE : "+str(dice[0])+"\n")
        logFile.write("IU : "+str(iu[0])+"\n")
        logFile.write("DICE_Patient : "+str(dice_patient[0])+"\n")
        logFile.write("IU_Patient : "+str(iu_patient[0])+"\n")
        logFile.close()
        print("Deep Learning, calcul Dice Iu : ")
        print('DICE_mean : '+str(dice[0]*100))
        print('IU_MEAN : '+str(iu[0]*100))
        print('DICE_Patient : '+str(dice_patient[0]*100))
        print('IU_Patient : '+str(iu_patient[0]*100))

        metric_results_slices = [{"Patient":s.patient_name, "idx":s.idx, "structure":s.structure,
                    "nbPixels":s.nbPixels[0], "tumor_size":s.positive[0],
                    "tp":s.true_positive[0], "fp":s.false_positive[0], "fn":s.false_negative[0],
                    "precision":s.precision[0], "recall":s.recall[0], "dice":s.dice[0], "iou": s.iou[0]} for s in data_train_test.patients[data_train_test.patient_test].slices.values()]
            
    return metric_results_patient, metric_results_slices



if __name__ == '__main__':
    
    #-----------------------------------------------#
    #           Initialisation                      #
    #-----------------------------------------------#
    
    #-------------get args from parser--------------#
    args = get_from_parser(verbose=True)

    #----------------init model path----------------#
    path_save_model_empty = '../save_model/unet_model_{}_{{}}_{}.h5'.format(args.experiment, args.structure)
    path_old_save_model_empty = '../save_model/unet_model_{}_{{}}_{}.h5'.format(args.ovassion_exp, args.structure)

    #--------------init results array---------------#
    results_patient = []
    results_slices = []

    #---------store experiment informations---------#
    store_experiment(args)
    
    for patient in args.patients :
        path_res = constant.RESULTS_DIR.format(args.experiment,patient)
        # set randomness
        np.random.seed(1)
        tf.random.set_seed(2)
        
        print("Patient : "+patient)
        args.patient = patient
        #-----------------------------------------------#
        #                 Get data                      #
        #-----------------------------------------------#
        print("GET DATA")
        
        data_train_test = data.get_data(patient, args)
        try:
            print(data_train_test.train_idx)
        except:
            print("NO train_idx")

        # -----------------------------------------------#
        #                 Create model                  #
        # -----------------------------------------------#
        print("MODEL INITIALISATION")

        #------------------from scratch-----------------#
        print("Create Model")
        # path_old_save_model = "save_model_pierre/unet_model_{}_200_epochs_4_bs_0.0001_lr_0_ctxt.h5".format(patient)
        # model = old_init_unet(path_old_save_model, args)
        path_old_save_model = path_old_save_model_empty.format(patient)
        model = init_unet(path_old_save_model, args)
        
        #-----------------load model--------------------#
        # path_to_old_model = path_save_model_empty.format(patient)
        # print("Get model :",path_to_old_model)
        # model = load_model(path_to_old_model,args)

        #----------load model to dropout model----------#
        # old_exp = "benchmark200"
        # if args.structure == "reinPatho":
        #     old_exp = "benchmarkKidney200"
        # if args.mode=="ovassion":
        #     old_exp+="_{}_ovassion".format(args.gap)

        # path_to_old_model = 'save_model/unet_model_{}_{}_{}.h5'.format(old_exp, patient, args.structure).format()
        # print("Get model from",old_exp,":",path_to_old_model)
        # model = load_model_to_dropout(path_to_old_model,args)

        #-----------------------------------------------#
        #                 Train model                   #
        #-----------------------------------------------#
        
        print("TRAIN MODEL")
        #---------------------train---------------------#
        model, history = train(model, data_train_test, args)
        training_plot_path = '../plots/training_plot_{}_{}_{}_{}epochs_{}lr_{}bs_{}_ctxt'.format(args.experiment,patient,args.structure,args.epochs,args.learning_rate,args.batch_size,args.context)
        save_training_plot(history, training_plot_path)
        
        #---------------------save----------------------#
        path_save_model = path_save_model_empty.format(patient)
        model.save(path_save_model)

        

        # -----------------------------------------------#
        #                 Predictions                   #
        # -----------------------------------------------#
        print("PREDICTIONS")
        

        # #-------------prediction------------------------#
        # # print("Prediction")
        # # test_predict(model, data_train_test, path_res)

        #------prediction with monte carlo dropout------#
        print("Monte-Carlo dropout prediction")
        test_predict_dropout(model, data_train_test, path_res)


        # -----------------------------------------------#
        #             Compute similarity                #
        # -----------------------------------------------#
        print("COMPUTE SIMILARITY")
        test_similarity(model, data_train_test, path_res)
        

        # -----------------------------------------------#
        #             Compute metrics                   #
        # -----------------------------------------------#
        print("COMPUTE METRICS")

        # -----------------get results-------------------#
        print("Get Predictions")
        get_predict(data_train_test, path_res)

        # --------------compute metrics------------------#
        print("Compute metrics")
        metric_results_patient, metric_results_slices = test_metrics(data_train_test, path_res)
        results_patient += metric_results_patient
        results_slices += metric_results_slices
    
    
    #-----------------------------------------------#
    #              Store results                    #
    #-----------------------------------------------#
    print("Store results csv")
    df_res = pd.DataFrame(results_patient).set_index(["Patient","structure"]).apply(pd.to_numeric)
    df_res.reset_index(inplace = True)
    structures = df_res.structure.unique()
    df_res_mean = df_res.copy()
    for structure in structures:
        df_res.loc[df_res.index.max()+1]= ["Mean",structure] + list(df_res_mean[df_res_mean.structure==structure].mean())
    df_res.set_index(["Patient","structure"], inplace = True)
    # df_res.loc["Mean"]= df_res.mean()
    df_res.to_csv(constant.CSV_FILE.format(args.experiment))
    df_res_slices = pd.DataFrame(results_slices).set_index(["Patient","idx","structure"]).apply(pd.to_numeric)
    df_res_slices.to_csv(constant.CSV_FILE_SLICES.format(args.experiment))
