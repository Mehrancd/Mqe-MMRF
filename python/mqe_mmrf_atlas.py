import numpy as np
import SimpleITK as sitk
import os
import sys
from deepbrain import Extractor
import matplotlib.pyplot as plt
from scipy.stats import norm
from func import myf 
import time
start_time = time.time()
print('Parameters list:')
# Load the image
params=['extractor', 'q-entropy' ,'alpha' ,'beta' ,'gamma' ,'NumberOfIteration']
params_val_default=np.array([0.125, -0.64, 0.1, 0.2, 0.1, 1])
try:
    inputDir =os.listdir('input') 
    outputDir ='output'
except IndexError:
    print("No enough parameters have been included (at least two parameters needed)")
    print("How to use:")
    print("docker run --network none -dit -v output_file:/output:rw -v input_file:/input -v [[parameters]:parameters:rw] --name mqe_mmrf_atlas  mrbrains18/csim")
    print("docker exec mqe_mmrf_atlas python3 /mrbrains18_csim/mqe_mmrf_atlas.py")
    print("Parameters: extractor, q-entropy ,alpha ,beta ,gamma ,NumberOfIteration")
    sys.exit()
try:
    parameters=sys.argv[1]
    for i in range(3,9):
        print(params[i-3])
        try:
            params_val_default[i-3]=parameters[i]
        except IndexError:
            print(params_val_default[i-3])
except IndexError:
    print("Some or all of parameters are in defualt")
t1Image = sitk.ReadImage(os.path.join('input',inputDir[0]))
t1Array = sitk.GetArrayFromImage(t1Image)
#define extractor for brain extraction
print('1-Brain extraction is started using Deep Brain extractor...')
ext = Extractor()
prob = ext.run(t1Array) 
# mask can be obtained as: where the best parameters was found for 0.125
prob[prob<params_val_default[0]]=0
prob[prob>=params_val_default[0]]=1
mask=prob
brain=np.multiply(t1Array,mask)
brain=np.absolute(brain)
MaxIn=brain.max()
MinIn=brain.min()
brain=(brain*255.0)/MaxIn
MaxIn=brain.max()
MinIn=brain.min()
brain=np.ceil(brain)
print('2-Image labeling is started using Tsallis entropy...')
q=params_val_default[1]
label_map=myf.Mqe(brain,q)
print('Image labeling done.')
print('3-Label correcting is started using MRF and AAR...')
#MMRF and ATLAS corrction for labelmap*******************************************************
Atlas = sitk.ReadImage('/mrbrains18_csim/data/Atlas.nii')
Atlas_label = sitk.ReadImage('/mrbrains18_csim/data/Atlas_label.nii')
Atlas_registered, label_registered =myf.Register_7DOF(sitk.GetImageFromArray(brain),Atlas,Atlas_label)
Atlas_registered = sitk.GetArrayFromImage(Atlas_registered)
muA,sigmaA=myf.label_statistic(Atlas_registered,sitk.GetArrayFromImage(label_registered))
alpha=params_val_default[2] 
beta=params_val_default[3] 
gamma=params_val_default[4] 
numberOfIterations=int(params_val_default[5])
n_weights=np.array([[1,1,1,1,2,1,1,1,1,1,2,1,2,0,2,1,2,1,1,1,1,1,2,1,1,1,1]]).transpose()
for it in range(numberOfIterations):
    muI,sigmaI=myf.label_statistic(brain,label_map)
    for k in range(brain.shape[0]): 
        if brain[k,:,:].max()==0:
            continue
        for i in range(brain.shape[1]): 
            for j in range(brain.shape[2]): 
                neighbors=myf.GetNeighbors(brain,[k,i,j])
                target=brain[k,i,j]
                if target==0 and neighbors[np.nonzero(neighbors)].size<10:
                    continue
                elif target==0:
                    target=neighbors.mean()
                n_labels=myf.GetNeighbors(label_map,[k,i,j])
                C1l=np.copy(n_labels)
                C1l[C1l!=1]=0
                C2l=np.copy(n_labels)
                C2l[C2l!=2]=0
                C3l=np.copy(n_labels)
                C3l[C3l!=3]=0
                C1=np.multiply(neighbors,C1l)
                C2=np.multiply(neighbors,C2l/2)
                C3=np.multiply(neighbors,C3l/3)
                mmrf=np.array([myf.MRF(C1,C2,C3,n_weights,target)])
                Image_weights=norm.pdf(target,loc=muI,scale=sigmaI).transpose()
                Image_weights[0,0]=Image_weights.min()
                Atlas_weights=norm.pdf(Atlas_registered[k,i,j],loc=muA,scale=sigmaA).transpose()
                Atlas_weights[0,0]=Atlas_weights.min()
                Ut=alpha*mmrf/mmrf.max()-beta*Image_weights/Image_weights.max()-gamma*Atlas_weights/Atlas_weights.max()
                new_label=np.argmin(Ut)
                label_map[k,i,j]=new_label
        print(np.floor((k/brain.shape[0])*100),'%')
print('100.0%')
print('the label correcting done. Elapsed time:')
elapsed_time = time.time() - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
for k in range(brain.shape[0]):
    for i in range(brain.shape[1]):
        for j in range(brain.shape[2]):
            if label_map[k,i,j]==1 :
                label_map[k,i,j]=3
            elif label_map[k,i,j]==2 or label_map[k,i,j]==3 :
                label_map[k,i,j]=label_map[k,i,j]-1 
resultImage = sitk.GetImageFromArray(label_map)
resultImage.CopyInformation(t1Image)
sitk.WriteImage(resultImage, os.path.join(outputDir,inputDir[0]+'_label.nii.gz'))
#output_file_name=outputDir+'/'+inputDir[0]+'_label_nii.gz'