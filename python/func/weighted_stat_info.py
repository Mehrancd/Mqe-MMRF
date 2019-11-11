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
