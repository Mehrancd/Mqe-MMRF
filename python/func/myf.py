# define Functions*************************************************************************
import numpy as np
from scipy.stats import norm
import SimpleITK as sitk
def GetNeighbors(img,Ind):
    p,r,c=Ind
    neighborhood=np.zeros([27,1])
    try:
        neighborhood[0] = img[p-1, r-1, c-1]
        neighborhood[1] = img[p-1, r,   c-1]
        neighborhood[2] = img[p-1, r+1, c-1]
    
        neighborhood[ 3] = img[p-1, r-1, c]
        neighborhood[ 4] = img[p-1, r,   c]#dist 1
        neighborhood[ 5] = img[p-1, r+1, c]
    
        neighborhood[ 6] = img[p-1, r-1, c+1]
        neighborhood[ 7] = img[p-1, r,   c+1]
        neighborhood[ 8] = img[p-1, r+1, c+1]

        neighborhood[ 9] = img[p, r-1, c-1]
        neighborhood[10] = img[p, r,   c-1]#dist 1
        neighborhood[11] = img[p, r+1, c-1]

        neighborhood[12] = img[p, r-1, c]#dist 1
        neighborhood[13] = 0 #img[p, r,   c] center
        neighborhood[14] = img[p, r+1, c]#dist 1

        neighborhood[15] = img[p, r-1, c+1]
        neighborhood[16] = img[p, r,   c+1]#dist 1
        neighborhood[17] = img[p, r+1, c+1]

        neighborhood[18] = img[p+1, r-1, c-1]
        neighborhood[19] = img[p+1, r,   c-1]
        neighborhood[20] = img[p+1, r+1, c-1]

        neighborhood[21] = img[p+1, r-1, c]
        neighborhood[22] = img[p+1, r,   c]#dist 1
        neighborhood[23] = img[p+1, r+1, c]

        neighborhood[24] = img[p+1, r-1, c+1]
        neighborhood[25] = img[p+1, r,   c+1]
        neighborhood[26] = img[p+1, r+1, c+1]
    except:
        p,r,c=Ind
    return neighborhood
# *************************************************************************************function label_statistic
def label_statistic(Image,label):
    muI=np.zeros([4,1])
    sigmaI=np.ones([4,1])
    muI[1]=Image[(Image>0) & (label==1)].mean()
    muI[2]=Image[(Image>0) & (label==2)].mean()
    muI[3]=Image[(Image>0) & (label==3)].mean()
    sigmaI[1]=np.sqrt(Image[(Image>0) & (label==1)].var())
    sigmaI[2]=np.sqrt(Image[(Image>0) & (label==2)].var())
    sigmaI[3]=np.sqrt(Image[(Image>0) & (label==3)].var())
    return muI,sigmaI
#******************************************************************mean and variance a vector for label and weight
def weighted_stat_info(V,W):
    #mu_n[3]=C3[np.nonzero(C3)].mean()*sum(1*(C3!=0))/sum(1*((n_labels==3)*(n_weights)))
    #sigma_n[1]=np.sqrt(C1[np.nonzero(C1)].var()*sum(1*(C1!=0))/sum(1*((n_labels==1)*(n_weights))))
    mu=0.0
    sigma=1
    sum_lw=sum((1*(V!=0))*W)
    V=np.multiply(V,W)
    #print(sum_lw)
    if sum_lw > 0 :
        mu=V[np.nonzero(V)].mean()*sum(1*(V!=0))/sum_lw
        sigma=np.sqrt(V[np.nonzero(V)].var()*sum(1*(V!=0))/sum_lw)
    if sigma<1:
        sigma=1        
    return mu,sigma
# ******************************************************************calculate the weights for each label besed on MRF
def MRF(V1,V2,V3,W,target):
    MRF_weights=np.zeros([4])
    mu=np.zeros([4])
    sigma=np.ones([4])
    mu[1], sigma[1]=weighted_stat_info(V1,W)
    mu[2], sigma[2]=weighted_stat_info(V2,W)
    mu[3], sigma[3]=weighted_stat_info(V3,W)
    V1size=V1[np.nonzero(V1)].size
    V2size=V1[np.nonzero(V2)].size
    V3size=V1[np.nonzero(V3)].size
    if target==0:
        MRF_weights=[1,2,3,4] #[background CSF GM WM] the min is the best
    else:
        MRF_weights[0]=4
        if V1size==0 and V2size==0 and V3size==0: #alone voxel
            MRF_weights=[1,2,3,4]
        elif V1size==0 and V2size==0 and V3size!=0: #Target has one neighbor in WM
            MRF_weights=[4,3,2,1]
        elif V1size==0 and V2size!=0 and V3size==0: #Target has one neighbor in GM
            MRF_weights=[4,2,1,3]
        elif V1size!=0 and V2size==0 and V3size==0: #Target has one neighbor in CSF
            MRF_weights=[4,1,2,3]
        elif V1size==0 and V2size!=0 and V3size!=0: #**********************************Target has two neighbor WM+GM  
            if target>=mu[3]+2*sigma[3] and V3size>2:
                MRF_weights=[4,3,2,1]
            elif target<mu[2]-2*sigma[2]: #decision is not clear maybe in GM or even CSF
                MRF_weights=[4,0,0,0]
            else:
                MRF_weights[1]=3 # is not in CSF
                #print(mu)
                #print(sigma)
                if (mu[2]+2*sigma[2])<(mu[3]-2*sigma[3]) and V3size>2 and V2size>2:
                    T=mu[2]+sigma[2]-((mu[2]+sigma[2])-(mu[3]-sigma[3]))/2 # may need consider
                    MRF_weights[2]=(target<=T)*1+(target>T)*2
                    MRF_weights[3]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[3],loc=mu[3],scale=sigma[3])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[3],scale=sigma[3])#that is inverse because min is used
                        MRF_weights[3]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[3],scale=sigma[3]) #that is inverse because finaly the min is used
                        MRF_weights[3]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
        elif V1size!=0 and V2size==0 and V3size!=0: #*********************************Target has two neighbor CSF+WM
            if target>mu[3]:
                MRF_weights=[4,2,3,1]
            elif target<(mu[3]-2*sigma[3]) and target>(mu[1]+2*sigma[1]): #decision is not clear in any tissue
                MRF_weights=[4,0,0,0]
            elif target<mu[1]:
                MRF_weights=[4,1,3,2]
            else:
                MRF_weights[2]=3 # is not in GM
                if (mu[1]+sigma[1])<(mu[3]-sigma[3]) and V1size>2 and V3size>2:
                    T=mu[1]+sigma[1]+((mu[3]+sigma[3])-(mu[1]-sigma[1]))/2
                    MRF_weights[1]=(target<=T)*1+(target>T)*2
                    MRF_weights[3]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[3],loc=mu[3],scale=sigma[3])
                    max2=norm.pdf(mu[1],loc=mu[1],scale=sigma[1])
                    if (max1/max2)>1:
                        MRF_weights[1]=(max1/max2)*norm.pdf(target,loc=mu[3],scale=sigma[3])#that is inverse because min is used
                        MRF_weights[3]=norm.pdf(target,loc=mu[1],scale=sigma[1])
                    else:
                        MRF_weights[1]=norm.pdf(target,loc=mu[3],scale=sigma[3]) #that is inverse because finaly the min is used
                        MRF_weights[3]=(max1/max2)*norm.pdf(target,loc=mu[1],scale=sigma[1])
        elif V1size!=0 and V2size!=0 and V3size==0: #************************************Target has two neighbor CSF+GM  
            if target>mu[2]+2*sigma[2] and V2size>2:#decision is not clear in any tissue
                MRF_weights=[4,0,0,0]
            elif target<mu[1]-2*sigma[1] and V1size>2:
                MRF_weights=[4,1,2,3]
            else:
                MRF_weights[3]=3 # is not in WM
                if (mu[1]+2*sigma[1])<(mu[2]-2*sigma[2]) and V1size>2 and V2size>2:
                    T=mu[1]+sigma[1]+((mu[2]-sigma[2])-(mu[1]+sigma[1]))/2
                    MRF_weights[1]=(target<=T)*1+(target>T)*2
                    MRF_weights[2]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[1],loc=mu[1],scale=sigma[1])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[1],scale=sigma[1])#that is inverse because min is used
                        MRF_weights[1]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[1],scale=sigma[1]) #that is inverse because finaly the min is used
                        MRF_weights[1]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
        else:# ******************************************************************************target has three neighbors
            if target>mu[3]+2*sigma[3] and V3size>2:
                MRF_weights=[4,3,2,1]
            elif target<mu[1]-2*sigma[1] and V1size>2:
                MRF_weights=[4,1,2,3]
            #elif target>=mu[1]-2*sigma[1] and target<mu[2]+sigma[2]: #target between CSF and GM
            elif target<mu[2]: #target between CSF and GM
                MRF_weights[3]=3 # is not in WM
                if (mu[1]+sigma[1])<(mu[2]-sigma[2]) and V1size>2 and V2size>2:
                    T=mu[1]+sigma[1]+((mu[2]-sigma[2])-(mu[1]+sigma[1]))/2
                    MRF_weights[1]=(target<=T)*1+(target>T)*2
                    MRF_weights[2]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[1],loc=mu[1],scale=sigma[1])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[1],scale=sigma[1])#that is inverse because min is used
                        MRF_weights[1]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[1],scale=sigma[1]) #that is inverse because finaly the min is used
                        MRF_weights[1]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
            else:#target between GM and WM
                MRF_weights[1]=3 # is not in CSF
                if (mu[2]+2*sigma[2])<(mu[3]-2*sigma[3]) and V3size>2 and V2size>2:
                    T=mu[2]+sigma[2]-((mu[2]+sigma[2])-(mu[3]-sigma[3]))/2
                    MRF_weights[2]=(target<=T)*1+(target>T)*2
                    MRF_weights[3]=(target<=T)*2+(target>T)*1
                else:
                    max1=norm.pdf(mu[3],loc=mu[3],scale=sigma[3])
                    max2=norm.pdf(mu[2],loc=mu[2],scale=sigma[2])
                    if (max1/max2)>1:
                        MRF_weights[2]=(max1/max2)*norm.pdf(target,loc=mu[3],scale=sigma[3])#that is inverse because min is used
                        MRF_weights[3]=norm.pdf(target,loc=mu[2],scale=sigma[2])
                    else:
                        MRF_weights[2]=norm.pdf(target,loc=mu[3],scale=sigma[3]) #that is inverse because finaly the min is used
                        MRF_weights[3]=(max1/max2)*norm.pdf(target,loc=mu[2],scale=sigma[2])
  
    return MRF_weights
#************************************Mqe**************************************************************************************
def Mqe(brain,q):
    
    #Histogram of the image
    Hist, bin_edges=np.histogram(brain,bins=range(0,257))
    #Hist=plt.hist(brain.ravel(), bins=range(1,int(MaxIn)+1))
    Hist[0]=0;
    norm_Hist=Hist/Hist.sum()  
    for i in range(1,Hist.size):
        if norm_Hist[i]>2.0e-5:
            first_bin=i
            break
    for i in range(Hist.size-1,1,-1):
        if norm_Hist[i]>2.0e-5:
            last_bin=i
            break
    Num_freq=np.sum(Hist)
# Mqe segmentation:calculating q-entropy to find the maximum entropy or t1 and t2 for thresholding
    maxim=-1.0e300
    for t1 in range(first_bin+10,last_bin-2):
        for t2 in range(t1+1,last_bin):
            num_freq1=np.sum(Hist[first_bin:t1+1])
            num_freq2=np.sum(Hist[t1+1:t2+1])
            num_freq3=np.sum(Hist[t2+1:last_bin]) 
            if (num_freq1==0 or num_freq2==0 or num_freq3)==0:
                continue
            weights1=Hist[first_bin:t1+1]/num_freq1
            weights2=Hist[t1+1:t2+1]/num_freq2
            weights3=Hist[t2+1:last_bin]/num_freq3                                   
            mu=np.array([np.average(range(first_bin,t1+1),weights=weights1), 
                     np.average(range(t1+1,t2+1),weights=weights2),
                     np.average(range(t2+1,last_bin),weights=weights3)])
            sigma=np.array([np.average((range(first_bin,t1+1)-mu[0])**2,weights=weights1), 
                     np.average((range(t1+1,t2+1)-mu[1])**2,weights=weights2),
                     np.average((range(t2+1,last_bin)-mu[2])**2,weights=weights3)])
            for i in range(3):
                if sigma[i]<1:
                    sigma[i]+=1
            pro=np.array([ num_freq1, num_freq2, num_freq3]/Num_freq)
            Xi=range(0,255)
            norm.pdf(3,loc=5,scale=1)
            GMM=pro[0]*norm.pdf(Xi,loc=mu[0],scale=sigma[0])+pro[1]*norm.pdf(Xi,loc=mu[1],scale=sigma[1])+pro[2]*norm.pdf(Xi,loc=mu[2],scale=sigma[2])
            SA=np.sum(np.power(GMM[first_bin:t1+1]/np.sum(GMM[first_bin:t1+1]),q))
            SA=(1-SA)/(q-1)
            SB=np.sum(np.power(GMM[t1+1:t2+1]/np.sum(GMM[t1+1:t2+1]),q))
            SB=(1-SB)/(q-1)
            SC=np.sum(np.power(GMM[t2+1:last_bin]/np.sum(GMM[t2+1:last_bin]),q))
            SC=(1-SC)/(q-1)
            topt=SA+SB+SC+(1-q)*(SA*SB+SA*SC+SB*SC)+(1-q)*(1-q)*SA*SB*SC;
            if(topt>=maxim):
                maxim=topt;
                T1=t1;
                T2=t2;          
    #print(T1,T2)
    label_map=np.zeros(brain.shape)
    label_map[(brain>0) & (brain<T1)]=1
    label_map[(brain>=T1) & (brain<T2)]=2
    label_map[brain>=T2]=3
    return label_map
#*****************************************************************************
def Register_7DOF(fixed,moving,label):
    samplingPercentage = 0.002
    R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMeanSquares()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingPercentage(samplingPercentage,sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetOptimizerAsRegularStepGradientDescent(0.05,.001,1500,0.5)
    tx=sitk.CenteredTransformInitializer(fixed, moving,sitk.ScaleVersor3DTransform())
    R.SetInitialTransform(tx,inPlace=True)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetNumberOfThreads(16)
    outTx = R.Execute(fixed, moving)
    print("-------")
    #print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))
    registered_image=sitk.Resample(moving, fixed, outTx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    registered_label=sitk.Resample(label, fixed, outTx, sitk.sitkLinear, 0.0, label.GetPixelID())
    type(registered_image)
    return registered_image, registered_label