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
