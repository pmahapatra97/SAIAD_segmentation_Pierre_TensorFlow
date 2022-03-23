# -*- coding: utf-8 -*-
import numpy as np
import os
from PIL import Image
import collections
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax
import constant

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
def dice_npyV2(listSeg, listSegVT, n_classes):
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
def iu_V2(listSeg, listSegVT, n_classes):
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

def printMeanDiceIU(meanDice, meanIU, logPath):
    #Ouverture du fichier en ecriture en mode Ecriture (ecrase ce qui existe deja)
    logFile = open(logPath,'w')
    #Dice
    logFile.write("DICE_mean : "+str(meanDice)+"\n")
    #IU
    logFile.write("IU_mean : "+str(meanIU)+"\n")
    #Fermeture
    logFile.close()
    
def saveColorSegmentation(out, nbImg, path, structure):
    im = np.array(out, dtype=np.uint8)

    for i in range(512):
        for j in range(512):
            if im[i][j][0] == 1 and im[i][j][1] == 1 and im[i][j][2] == 1: #tumeur
                if structure == 'reinPatho' :
                    im[i][j][0]= 90
                    im[i][j][1]= 52
                    im[i][j][2]= 41
                else :
                    im[i][j][0]= 72
                    im[i][j][1]= 119
                    im[i][j][2]= 72
    img = Image.fromarray(im)
    img.save(path+'{}.png'.format(nbImg))
                
def crfPostProcessing(imgScanner, imgSoftmax, nbImg, patient, structure, resultpath):
    
    nblabelPixel = np.count_nonzero(imgSoftmax[1]>=0.5)    
    imgSoftmax = imgSoftmax.squeeze()
                
    #imgSoftmax = imgSoftmax.transpose((2, 0, 1))
 
    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    # unary = softmax_to_unary(processed_probabilities)
    unary = unary_from_softmax(imgSoftmax)
    
    #arguments : tailles de limage et nombre de labels
    d = dcrf.DenseCRF(imgScanner.shape[0] * imgScanner.shape[1],2)
    d.setUnaryEnergy(unary)
    
    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations

    
    if(structure == "tumeur"):
        feats = create_pairwise_gaussian(sdims=(10,10), shape=imgScanner.shape[:2])
    else : 
        feats = create_pairwise_gaussian(sdims=(2,2), shape=imgScanner.shape[:2])
    
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    
    if(structure == "tumeur"):
        feats = create_pairwise_bilateral(sdims=(8, 8), schan=(10, 10, 10),
                                       img=imgScanner, chdim=2)
    else : 
        feats = create_pairwise_bilateral(sdims=(3, 3), schan=(5, 5, 5),
                                       img=imgScanner, chdim=2)
    d.addPairwiseEnergy(feats, compat=10,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)
    
    Q = d.inference(5)
    #print(len(np.argmax(Q, axis=0)))
    
    
    #Segmentation apres CRF
    res = np.argmax(Q, axis=0).reshape((imgScanner.shape[0], imgScanner.shape[1]))
                
    res = np.array(res, dtype=np.uint8)
    
    nblabelPixelFinal = np.count_nonzero(res == 1)
    
    #Si les CRF ont enleve beaucoups de pixels labelise, c'est qu'il a eneleve une trop grosse 
    #partie de l'objet, Donc on realise un crf sans pairwise gaussian
    if nblabelPixel/2 > nblabelPixelFinal : 
        d = dcrf.DenseCRF(imgScanner.shape[0] * imgScanner.shape[1],2)
        d.setUnaryEnergy(unary)
        if(structure == "tumeur"):
            feats = create_pairwise_bilateral(sdims=(8, 8), schan=(10, 10, 10),
                                           img=imgScanner, chdim=2)
        else : 
            feats = create_pairwise_bilateral(sdims=(3, 3), schan=(5, 5, 5),
                                           img=imgScanner, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                             kernel=dcrf.DIAG_KERNEL,
                             normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).reshape((imgScanner.shape[0], imgScanner.shape[1]))
        res = np.array(res, dtype=np.uint8)
        
        nblabelPixelFinal = np.count_nonzero(res == 1)
    
    kernel = np.ones((2,2),np.uint8)
    res = cv2.morphologyEx(res,cv2.MORPH_CLOSE,kernel, iterations = 1)
    
    res = cv2.medianBlur(res,5)
    
    res2 = Image.fromarray(res)
    res2.save(resultpath.format(patient)+'/{}.png'.format(nbImg))
    res2 = res2.convert('RGB')
    saveColorSegmentation(res2 ,nbImg, resultpath.format(patient)+'/color/', structure)

    return res


#structure "tumeur" ou "reinPatho"
def postProcessing(patient, structure, experiment, ovassion_training, gap=None):
    listSegVT = collections.OrderedDict()
    # if ovassion_training:
    #     logPathDL = constant.RESULTS_OVA_DIR.format(patient)+'/logs/resultDiceIU.txt'
    #     logPathCRF =constant.RESULTS_OVA_DIR.format(patient)+'/logs/resultDiceIU_CRF.txt'
    #     resultSegPath = constant.RESULTS_OVA_DIR.format(patient)+'/classes'
    #     resultSegPathCRF = constant.RESULTS_OVA_DIR.format(patient)+'/CRF'
    # else:
    print("postProcessing result path: ",constant.RESULTS_DIR.format(experiment,patient))
    logPathDL = constant.RESULTS_DIR.format(experiment,patient)+'/logs/resultDiceIU.txt'
    logPathCRF =constant.RESULTS_DIR.format(experiment,patient)+'/logs/resultDiceIU_CRF.txt'
    resultSegPath = constant.RESULTS_DIR.format(experiment,patient)+'/classes'
    resultSegPathCRF = constant.RESULTS_DIR.format(experiment,patient)+'/CRF'


    listDL = collections.OrderedDict()
    listCRF = collections.OrderedDict()
    
    listFile = os.listdir(resultSegPath.format(patient))
    lenImg = len([s for s in listFile if "png" in s])
    
    
    
    testSlices = [i for i in range(1,lenImg+1)]
    
    if ovassion_training:
        trainingSlices = []
        for i in range(1,lenImg+1,gap+1):
            trainingSlices.append(i)
        
        testSlices = list(set(testSlices) - set(trainingSlices))
        
    #Pour chaque segmentations calculees
    for idx in testSlices:
        
        #Ouverture de la seg calculee
        out1 = Image.open(resultSegPath+'/{}.png'.format(idx)).convert('L')
        segOut = np.array(out1, dtype=np.uint8)
        listDL[idx] = segOut

        # if idx == 8:
        #     print(segOut[256])

        #Ouverture de la seg verite terrain
        seg1 = Image.open(constant.SEG_DIR_TEST.format(patient,structure)+'{}.png'.format(idx)).convert('L')
        segVT = np.array(seg1, dtype=np.uint8)
    
        listSegVT[idx] = segVT

        #Ouverture de l'image scanner 
        scan = Image.open(constant.SCANNER_DIR.format(patient)+'{}.jpg'.format(idx)).convert('RGB')
        scan = np.array(scan, dtype=np.uint8)
        
        #Ouverture de la map de probabilite 
        segProb1 = Image.open(resultSegPathCRF+'/Map/{}_1.tiff'.format(idx))
        segProb1 = np.array(segProb1)
        segProb2 = Image.open(resultSegPathCRF+'/Map/{}_2.tiff'.format(idx))
        segProb2 = np.array(segProb2)
        segProb = np.zeros((2,512,512))
        segProb[0] = segProb1
        segProb[1] = segProb2
        
        res = crfPostProcessing(scan, segProb, idx, patient, structure, resultSegPathCRF)
        
        
        res = Image.open(resultSegPathCRF+'/{}.png'.format(idx)).convert('L')
        res = np.array(res, dtype=np.uint8)
        
        listCRF[idx] = res
        
        idx +=1
        
    
    dice = dice_npyV2(listDL,listSegVT,2)[1]
    iu = iu_V2(listDL,listSegVT,2)[1]    
    
    printMeanDiceIU(dice, iu, logPathDL)
    print("Deep Learning, calcul Dice Iu : ")
    print('DICE_mean : '+str(dice*100))
    print('IU_MEAN : '+str(iu*100))
        
    
    dice = dice_npyV2(listCRF,listSegVT,2)[1]
    iu = iu_V2(listCRF,listSegVT,2)[1]
    
    printMeanDiceIU(dice, iu, logPathCRF)
    
    print("Deep Learning + CRF, calcul Dice IU : ")
    print('DICE_mean CRF : '+str(dice*100))
    print('IU_MEAN CRF : '+str(iu*100))
    
if __name__ == '__main__':       
    
    #patients = ['SAIAD_01','SAIAD_02','SAIAD_02Bis','SAIAD_04','SAIAD_05','SAIAD_07','SAIAD_09',
                #'SAIAD_10','SAIAD_11','SAIAD_12','SAIAD_13','SAIAD_14','SAIAD_15','SAIAD_15Bis']
    patients = ['SAIAD_01']
    ovassion_training=False
    for patient in patients:
        postProcessing(patient, "tumeur",  ovassion_training=ovassion_training)
