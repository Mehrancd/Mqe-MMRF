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
    print(T1,T2)
    label_map=np.zeros(brain.shape)
    label_map[(brain>0) & (brain<T1)]=1
    label_map[(brain>=T1) & (brain<T2)]=2
    label_map[brain>=T2]=3
    return label_map
