def label_statistic(Image,label):
    muI=np.zeros([4,1])
    sigmaI=np.ones([4,1])
    muI[1]=Image[label==1].mean()
    muI[2]=Image[label==2].mean()
    muI[3]=Image[label==3].mean()
    sigmaI[1]=np.sqrt(Image[label==1].var())
    sigmaI[2]=np.sqrt(Image[label==2].var())
    sigmaI[3]=np.sqrt(Image[label==3].var())
    return muI,sigmaI
