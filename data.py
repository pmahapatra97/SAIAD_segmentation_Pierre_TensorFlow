# -*- coding: utf-8 -*-
import os
from PIL import Image, ImageEnhance
import numpy as np
import constant
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
from itertools import starmap

def precision_metric(tp, fp):
    def p(tp,fp):
        if tp==0 and fp==0:
            return 1
        else:
            return tp/(tp+fp)
    return np.array(list(starmap(p,zip(tp,fp))))
def recall_metric(tp, fn):
    def r(tp,fn):
        if tp==0 and fn==0:
            return 1
        else:
            return tp/(tp+fn)
    return np.array(list(starmap(r,zip(tp,fn))))

def dice_metric(tp, fp, fn):
    def d(tp,fp,fn):
        if tp==0 and fp ==0 and fn == 0:
            return 1
        else:
            return 2*tp/(2*tp+fn+fp)
    return np.array(list(starmap(d,zip(tp,fp,fn))))

def iou_metric(tp, fp, fn):
    def iou(tp,fp,fn):
        if tp==0 and fp ==0 and fn == 0:
            return 1
        else :
            return tp/(tp+fp+fn)
    return np.array(list(starmap(iou,zip(tp,fp,fn))))

def all_metrics(tp, fp, fn):
    return (precision_metric(tp, fp),recall_metric(tp, fn),dice_metric(tp, fp, fn),iou_metric(tp, fp, fn))





class Slice():
    def __init__(self, structure, patient_name, idx, experiment=None):
        self.structure = structure
        self.patient_name = patient_name
        self.idx = idx
        self.load_slice()
        self.gt = True
        self.pred = False

    def load_slice(self):
        #Scanner Image
        im = Image.open(constant.SCANNER_PREPROC_DIR.format(self.patient_name)+'{}.jpg'.format(self.idx)).convert('L')
        #Pre-treatment: sharpness enhancement
        enhancer_sharp = ImageEnhance.Sharpness(im)
        im = enhancer_sharp.enhance(2)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_ / 255.0
        in_ = in_[..., np.newaxis]
        
        #GT Segmentation
        im_seg = Image.open(constant.SEG_DIR.format(self.patient_name,self.structure)+'{}.png'.format(self.idx)).convert('L')
        if self.structure=="tumeur_reinPatho":
            label = tf.keras.utils.to_categorical(im_seg,num_classes=3)
        else:
            label = np.array(im_seg, dtype=np.float32)
            label = label[..., np.newaxis]

        # #Ouverture de la seg calculee
        # if experiment is not None:
        #     out1 = Image.open(constant.RESULTS_DIR.format(experiment,patient)+'/classes/{}.png'.format(idx)).convert('L')
        #     segOut = np.array(out1, dtype=np.float33)
        
        # Store
        self.scan = in_
        self.gt_seg = label
        self.pred_seg = None

        #Compute first features
        ax = np.array([1,2]) if self.structure == "tumeur_reinPatho" else [0]
        self.nbPixels =np.repeat(int(self.gt_seg.size/self.gt_seg.shape[2]),self.gt_seg.shape[2])[ax]
        self.positive = np.sum(self.gt_seg == 1, axis = (0,1))[ax]
        self.negative = np.sum(self.gt_seg == 0, axis = (0,1))[ax]
        if self.structure == "tumeur_reinPatho":
            self.nbPixels  = np.concatenate([self.nbPixels,[np.sum(self.nbPixels)]])
            self.positive = np.concatenate([self.positive,[np.sum(self.positive)]])
            self.negative = np.concatenate([self.negative,[np.sum(self.negative)]])
        
    def set_test(self):
        self.gt = False

    def set_train(self):
        self.gt = True

    def set_pred(self,pred_seg):
        self.pred_seg =pred_seg
        self.pred = True
        self.compute_metrics()

    def set_std(self, pred_std):
        self.pred_std =pred_std
        self.pred = True
        self.compute_std_metrics()


    def compute_metrics(self):
        ax = np.array([1,2]) if self.structure == "tumeur_reinPatho" else [0]
        self.true_positive  = np.sum(np.logical_and(self.pred_seg == 1, self.gt_seg == 1), axis = (0,1))[ax]
        self.false_positive = np.sum(np.logical_and(self.pred_seg == 1, self.gt_seg == 0), axis = (0,1))[ax]
        self.false_negative = np.sum(np.logical_and(self.pred_seg == 0, self.gt_seg == 1), axis = (0,1))[ax]
        if self.structure == "tumeur_reinPatho":
            self.true_positive  = np.concatenate([self.true_positive,[np.sum(self.true_positive)]])
            self.false_positive = np.concatenate([self.false_positive,[np.sum(self.false_positive)]])
            self.false_negative = np.concatenate([self.false_negative,[np.sum(self.false_negative)]])
        self.precision, self.recall, self.dice, self.iou = all_metrics(self.true_positive,self.false_positive,self.false_negative)

    def compute_std_metrics(self):
        self.absolute_std = np.sum(self.pred_std)
        self.relative_std = self.absolute_std/(1+np.sum(self.pred_seg))
        

class Patient:
    def __init__(self, structure, name):
        self.name = name
        self.structure = structure
        self.load_patient()
    
    def load_patient(self):
        listFile = [s for s in os.listdir(constant.SCANNER_PREPROC_DIR.format(self.name)) if ".jpg" in s]
        self.nbSlices = len(listFile)
        listIdx = map(lambda file:int(os.path.splitext(file)[0]),listFile)
        self.slices={}
        for idx in listIdx:
            self.slices[idx] = Slice(self.structure,self.name, idx)
        self.nbPixels = np.sum([s.nbPixels for s in self.slices.values()], axis=0)
        self.positive = np.sum([s.positive for s in self.slices.values()], axis=0)
        self.negative = np.sum([s.negative for s in self.slices.values()], axis=0)
        self.positive_r = self.positive/self.nbPixels
        self.negative_r = self.negative/self.nbPixels
        
    def compute_metrics(self):
        self.true_positive = np.sum([s.true_positive for s in self.slices.values() if not s.gt], axis = 0)
        self.false_positive = np.sum([s.false_positive for s in self.slices.values() if not s.gt], axis = 0)
        self.false_negative = np.sum([s.false_negative for s in self.slices.values() if not s.gt], axis = 0)
        self.true_positive_patient = np.sum([s.true_positive for s in self.slices.values()], axis =0)
        self.false_positive_patient = np.sum([s.false_positive for s in self.slices.values()], axis =0)
        self.false_negative_patient = np.sum([s.false_negative for s in self.slices.values()], axis =0)
        self.precision, self.recall, self.dice, self.iou = all_metrics(self.true_positive,self.false_positive,self.false_negative)
        self.precision_patient, self.recall_patient, self.dice_patient, self.iou_patient = all_metrics(self.true_positive_patient,self.false_positive_patient,self.false_negative_patient)
        print("METRICS")
        print(self.positive)
        print(self.negative)
        print(self.positive_r)
        print(self.negative_r)
        print(self.true_positive)
        print(self.false_positive)
        print(self.false_negative)
        print(self.true_positive_patient)
        print(self.false_positive_patient)
        print(self.false_negative_patient)
        print(self.dice)
        print(self.dice_patient)


    def get_slice(self, idx, seg_type="ground_truth"):
        if seg_type == "ground_truth":
            return (self.slices[idx].scan, self.slices[idx].gt_seg)
        elif seg_type == "prediction":
            return (self.slices[idx].scan, self.slices[idx].pred_seg)
        else:
            raise ValueError("seg_type can only take these values: ground_truth, prediction")

    def get_slices(self, slices_idx, seg_type="ground_truth"):
        return tuple(np.array(l) for l in zip(*[self.get_slice(idx, seg_type) for idx in slices_idx]))

    def get_patient(self, seg_type="ground_truth"):
        return self.get_slices(range(1,self.nbSlices+1), seg_type)

    def set_similarity(self, sim_mat):
        self.sim_mat = sim_mat

class SaiadData:
    def __init__(self,structure, patient_test, mode="normal"):
        self.structure = structure
        self.patient_test = patient_test
        self.mode = mode
        self.load_data()
        self.set_test(patient_test)

    def load_data(self):
        self.patients = {patient: Patient(self.structure, patient) for patient in constant.BASE_CASE_PATIENT_NAMES}
        self.nbSlices = np.sum([p.nbSlices for p in self.patients.values()])
        self.nbTrain = np.sum([self.patients[patient].nbSlices for patient in constant.BASE_CASE_PATIENT_NAMES if patient != self.patient_test])

    def set_test(self, patient_test):
        self.patient_test = patient_test
        [s.set_test() for s in self.patients[self.patient_test].slices.values()]

    def get_patient_slices(self, patients, seg_type="ground_truth"):
        return tuple(np.concatenate(l) for l in zip(*[self.patients[patient].get_patient(seg_type) for patient in patients]))

    def get_patient_test(self, seg_type="ground_truth"):
        return self.get_patient_slices([self.patient_test], seg_type)

    def get_slices(self, patient, slices_idx, seg_type="ground_truth"):
        return self.patients[patient].get_slices(slices_idx, seg_type)

    def get_train(self, seg_type="ground_truth"):
        patients = [patient for patient in constant.BASE_CASE_PATIENT_NAMES if patient != self.patient_test]
        return self.get_patient_slices(patients, seg_type)

    def get_test(self, seg_type="ground_truth"):
        return self.get_patient_slices([self.patient_test], seg_type)

    def get_train_val_gen(self,batch_size, augmentation = True, enhanced = True):
        patients = [patient for patient in constant.BASE_CASE_PATIENT_NAMES if patient != self.patient_test]
        print(patients)
        return create_generator(patients=patients, structure=self.structure, augmentation=augmentation, enhanced=enhanced,batch_size=batch_size)
    def get_test_gen(self,batch_size=1, enhanced = True):
        return create_generator(patients=[self.patient_test], structure=self.structure, augmentation=False, enhanced=enhanced,batch_size=batch_size, validation_split=0, shuffle=False)



class SaiadData_ovassion(SaiadData):
    def __init__(self,structure, patient_test, mode ="ovassion", path_res = None, selection="gap", nbSelected=None, gap=None, shift=0):
        self.path_res = path_res
        self.gap = gap
        self.selection = selection
        self.nbSteps = int((gap+1)/2) if gap is not None else None
        self.nbSelected = nbSelected
        self.shift = shift
        super().__init__(structure, patient_test, mode)
        
    def load_data(self):
        self.patients = {self.patient_test:Patient(self.structure, self.patient_test)}
        self.nbSlices = self.patients[self.patient_test].nbSlices
        self.load_idx()
        
    def load_idx(self):
        np.random.seed(1)
        if self.selection=="gap":
            if self.shift is None:
                self.shift = ((self.nbSlices-1)%(self.gap+1))//2
            self.train_idx = [idx for idx in range(1,self.nbSlices+1) if (idx-1-self.shift) % (self.gap+1) == 0]
        elif self.selection=="number":
            if self.shift is None:
                self.shift = 0
            self.train_idx = list(np.linspace(1+self.shift, self.nbSlices, self.nbSelected, dtype=int))
        elif self.selection == "random":
            self.train_idx = list(np.sort(np.random.choice(range(1,self.nbSlices+1), self.nbSelected, replace=False)))
        elif self.selection == "active":
            print("Get sim-matrix:",self.path_res+'sim.npy')
            simm = np.load(self.path_res+'sim.npy')
            self.train_idx = get_subset(simm, self.nbSelected, self.gap//2)
            print(self.train_idx)
        self.nbTrain = len(self.train_idx)
        self.test_idx  = [idx for idx in range(1,self.nbSlices+1) if idx not in self.train_idx]

    def set_test(self, patient_test):
        self.patient_test = patient_test
        [s.set_test() for s in self.patients[self.patient_test].slices.values() if s.idx in self.test_idx]
    
    def get_train(self, seg_type="ground_truth"):
        # print(self.train_idx)
        return self.get_slices(self.patient_test, self.train_idx, seg_type)
        
    def get_test(self, seg_type="ground_truth"):
        return self.get_slices(self.patient_test, self.test_idx, seg_type)

    def step_idx(self, step):
        return [idx for idx in range(1,self.nbSlices+1) if (idx-1-self.shift) % (self.gap+1) in [step, self.gap+1-step]]
    
    def get_step(self, step, seg_type="ground_truth"):
        # print(self.step_idx(step))
        return self.get_slices(self.patient_test, self.step_idx(step), seg_type)

    def set_pred(self, step, pred_segs):
        for iidx, pred_seg in zip(self.step_idx(step), pred_segs):
            self.patients[self.patient_test].slices[iidx].set_pred(pred_seg)
    
    def get_train_val_gen(self,batch_size, augmentation = True, enhanced = True):
        patients = [self.patient_test]
        return create_generator(patients=patients, slices=self.train_idx, structure=self.structure, augmentation=augmentation, enhanced=enhanced,batch_size=batch_size, validation_split=0)
    def get_test_gen(self,batch_size=1, enhanced = True):
        return create_generator(patients=[self.patient_test], structure=self.structure, augmentation=False, enhanced=enhanced,batch_size=batch_size, validation_split=0, shuffle=False)
  

    def set_shift(self,shift):
        self.shift = shift
        self.load_idx()


def get_data(patient_test, args):


    if args.mode=="normal":
        return SaiadData(args.structure, patient_test)
    else:
        return SaiadData_ovassion(args.structure, patient_test, args.mode, constant.RESULTS_DIR.format(args.ovassion_exp,patient_test), args.selection, args.nbSelected, args.gap, args.shift )


def repres(simm, idx):
    return max(simm[:,idx])

def sub_sim(simm, subset_ind):
    return sum([repres(simm, idx) for idx in subset_ind])

def find_first(subset_ind):
    for i in range(len(subset_ind)):
        if subset_ind[i]!=i:
            return i
    return len(subset_ind)
def get_subset(simm, n_samples,gap=None):
    subset_ind = []
    subset_taken = []
    n = simm.shape[0]
    for k in range(n_samples):
        if len(subset_taken)==n:
            break
        i_c = find_first(subset_taken)
        score_c = sub_sim(simm, subset_ind+[i_c])
        i=i_c+1
        while i<n:
            score = 0
            if i not in subset_taken:
                score_new = sub_sim(simm, subset_ind+[i])
                if score_new > score_c:
                    score_c = score_new
                    i_c = i
            i+=1
        subset_ind = subset_ind + [i_c]
        if gap is not None:
            subset_taken = list(set(subset_taken + [i for i in range(i_c-gap,i_c+gap+1) if i>=0 and i < n]))
        else:
            subset_taken = list(set(subset_taken + [i_c]))
        subset_ind.sort()
        subset_taken.sort()
        
    return [s+1 for s in subset_ind]

def create_generator(patients=constant.BASE_CASE_PATIENT_NAMES, 
    slices=None, structure="tumeur", augmentation=True, enhanced=True,
    batch_size=1, validation_split=0.1,shuffle = True):
    print(patients)
    image_list = []
    mask_list = []
    for patient in patients:
        listFile = [s for s in os.listdir(constant.SCANNER_DIR.format(patient)) if ".jpg" in s]
        path = constant.SCANNER_PREPROC_DIR.format(patient)
        listIdx = list(map(lambda file:int(os.path.splitext(file)[0]),listFile))
        listIdx.sort()
        listPathImage = [os.path.abspath(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(idx)) for idx in listIdx]
        image_list.append(pd.DataFrame(listPathImage,columns=['filename'],index=listIdx))
        if structure == "tumeur_reinPatho":
            listPathMask = [os.path.abspath(constant.SEG_DIR_OHE.format(patient,structure)+'{}.png'.format(idx)) for idx in listIdx]
            color_mode = "rgb"
        else:
            listPathMask = [os.path.abspath(constant.SEG_DIR.format(patient,structure)+'{}.png'.format(idx)) for idx in listIdx]
            color_mode = "grayscale"
        mask_list.append(pd.DataFrame(listPathMask,columns=['filename'],index=listIdx))
    if slices is None:
        train_image_df = pd.concat(image_list)
        train_mask_df = pd.concat(mask_list)
    else:
        train_image_df = image_list[0].loc[slices]
        train_mask_df = mask_list[0].loc[slices]
        val_slices = [i for i in listIdx if i not in slices]
        val_image_df = image_list[0].loc[val_slices]
        val_mask_df = mask_list[0].loc[val_slices]
    # we create two instances with the same arguments
    if enhanced:
        def enhance(im):
            img = Image.fromarray(im[:,:,0]).convert('L')
            enhancer_sharp = ImageEnhance.Sharpness(img)
            im_e = enhancer_sharp.enhance(2)
            im_e = np.array(im_e,dtype=np.float64)[...,np.newaxis]
            return im_e
    else: 
        def enhance(im):
            return im
    if augmentation:
        data_gen_aug_all = dict(
                         #featurewise_center=True,
                         #featurewise_std_normalization=True,
                         #preprocessing_function = ,
                         rotation_range=5,
                         width_shift_range=0.05,
                         height_shift_range=0.05,
                         zoom_range=0.1,
                         # horizontal_flip=True
        )
        data_gen_aug_scan = dict(brightness_range=(0.75,1.5))
    else:
        data_gen_aug_all = dict()
        data_gen_aug_scan = dict()
    # if structure == "tumeur_reinPatho":
    #     print("define")
    #     def p_function(im):
    #         print(im.shape)
    #         im_c = tf.keras.utils.to_categorical(im,num_classes=3)
    #         print(im_c)
    #         return im_c

    # else:
    #     print("None")
    #     p_function=None
    # print(p_function)
    data_gen_scan = dict(rescale=1./255, preprocessing_function = enhance)
    data_gen_all = dict(validation_split=validation_split, dtype=np.float32)

    image_datagen = ImageDataGenerator(**data_gen_all,**data_gen_aug_all,**data_gen_scan,**data_gen_aug_scan)
    mask_datagen = ImageDataGenerator(**data_gen_all,**data_gen_aug_all)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    train_image_generator = image_datagen.flow_from_dataframe(
        #dataframe=train_image_df.head(50).tail(1),
        dataframe=train_image_df,
        subset="training",
        batch_size=batch_size,
        class_mode=None,
        data_format="channel_last",
        target_size = (512,512),
        color_mode = "grayscale",
        seed=seed,shuffle=shuffle)
    train_mask_generator = mask_datagen.flow_from_dataframe(
        #dataframe=train_mask_df.head(50).tail(1),
        dataframe=train_mask_df,
        subset="training",
        batch_size=batch_size,
        class_mode=None,
        data_format="channel_last",
        target_size = (512,512),
        color_mode = color_mode,
        seed=seed,shuffle=shuffle)
    train_generator = zip(train_image_generator, train_mask_generator)
    if slices is None:
        print("slices is None")
        val_image_generator = image_datagen.flow_from_dataframe(
            #dataframe=val_image_df.head(50).tail(1),
            dataframe=train_image_df,
            subset="validation",
            batch_size=batch_size,
            class_mode=None,
            data_format="channel_last",
            target_size = (512,512),
            color_mode = "grayscale",
            seed=seed,shuffle=shuffle)
        val_mask_generator = mask_datagen.flow_from_dataframe(
            #dataframe=val_mask_df.head(50).tail(1),
            dataframe=train_mask_df,
            subset="validation",
            batch_size=batch_size,
            class_mode=None,
            data_format="channel_last",
            target_size = (512,512),
            color_mode = color_mode,
            seed=seed,shuffle=shuffle)
        val_generator = zip(val_image_generator, val_mask_generator)
    else:
        print("slices :"+str(slices))
        data_gen_all = dict(validation_split=0, dtype=np.float32)
        val_image_datagen = ImageDataGenerator(**data_gen_all,**data_gen_aug_all,**data_gen_scan,**data_gen_aug_scan)
        val_mask_datagen = ImageDataGenerator(**data_gen_all,**data_gen_aug_all)
        val_image_generator = val_image_datagen.flow_from_dataframe(
            #dataframe=val_image_df.head(50).tail(1),
            dataframe=val_image_df,
            subset="training",
            batch_size=batch_size,
            class_mode=None,
            data_format="channel_last",
            target_size = (512,512),
            color_mode = "grayscale",
            seed=seed,shuffle=shuffle)
        val_mask_generator = val_mask_datagen.flow_from_dataframe(
            #dataframe=val_mask_df.head(50).tail(1),
            dataframe=val_mask_df,
            subset="training",
            batch_size=batch_size,
            class_mode=None,
            data_format="channel_last",
            target_size = (512,512),
            color_mode = color_mode,
            seed=seed,shuffle=shuffle)
        val_generator = zip(val_image_generator, val_mask_generator)
    return train_generator, val_generator

    
# def get_slice_scan_seg(patient, data, gt_segmentation, structure, idx, nb_images, context=0):
    
#     #Image Scanner
#     im = Image.open(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(idx)).convert('L')

#     #Pre-traitement augmentation de la nettete
#     enhancer_sharp = ImageEnhance.Sharpness(im)
#     im = enhancer_sharp.enhance(2)
#     in_ = np.array(im, dtype=np.float32)
#     in_ = in_ / 255.0
#     in_ = in_[..., np.newaxis]
#     for ctx in range(context):
#         im_inf = Image.open(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(max(idx-1,1))).convert('L')
#         # print(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(max(idx-1,1)))
#         im_inf = ImageEnhance.Sharpness(im_inf).enhance(2)
#         in_inf = (np.array(im_inf, dtype=np.float32)/255.0)[..., np.newaxis]

#         im_sup = Image.open(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(min(idx+1,nb_images))).convert('L')
#         # print(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(min(idx+1,nb_images)))
#         im_sup = ImageEnhance.Sharpness(im_sup).enhance(2)
#         in_sup = (np.array(im_inf, dtype=np.float32)/255.0)[..., np.newaxis]

#         in_ = np.concatenate([in_inf,in_,in_sup],-1)
#     data.append(in_)

    
#     #Segmentation VT
#     im_seg = Image.open(constant.SEG_DIR.format(patient,structure)+'{}.png'.format(idx)).convert('L')
#     label = np.array(im_seg, dtype=np.float32)
#     label = label[..., np.newaxis]
#     gt_segmentation.append(label)

#     return data, gt_segmentation

# def get_data(patient_test, structure, mode, gap = 4, context=0):

#     if mode == "normal":
#         data_idx = [[] for x in range(2)]
#         for patient in constant.BASE_CASE_PATIENT_NAMES:
#             listFile = os.listdir(constant.SCANNER_PREPROC_DIR.format(patient))
#             listFile.sort()
#             nb_images = len([s for s in listFile if "jpg" in s]) 
#             if patient != patient_test:
#                 data_idx[0] += [(patient,idx) for idx in range(1,nb_images+1)]
#             else:
#                 data_idx[1] += [(patient,idx) for idx in range(1,nb_images+1)]
#     else:
#         listFile = os.listdir(constant.SCANNER_PREPROC_DIR.format(patient_test))
#         listFile.sort()
#         nb_images = len([s for s in listFile if "jpg" in s]) 

#         if mode == "ovassion":
#             data_idx = []
#             data_idx += [[(patient_test,idx) for idx in range(1,nb_images+1) if idx % (gap+1) == 1]]
#             data_idx += [[(patient_test,idx) for idx in range(1,nb_images+1) if idx % (gap+1) != 1]]
            
#         elif mode == "ovassion_r":
#             data_idx = [[(patient_test,idx) for idx in range(1,nb_images+1)
#                                 if (idx-1) % (gap+1) in [step, gap+1-step]] 
#                                 for step in range(int((gap+1)/2)+1)]
#     # print(data_idx)

#     data = []
#     for data_id in data_idx:
#         img, gt_segmentation = [], []
#         for patient, idx in data_id:
#             img, gt_segmentation = get_slice_scan_seg(patient, img, gt_segmentation, structure, idx, nb_images, context) 
#         data.append((np.array(img).astype('float32'), np.array(gt_segmentation).astype('float32')))
    
#     return data, data_idx
