#import numpy as np
from scipy.stats import norm
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
