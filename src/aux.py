import numpy as np
import matplotlib.pyplot as plt
# generate sample data from a micture of Gaussians
def gauss_mix(n=10000,g=20,sigma=0.01, d=2,rand_sig=None):
    gaus=[]
    for _ in range(g):
        if rand_sig is None:
            sig = sigma
        else:
            sig = sigma*(np.random.rand()*rand_sig+1/rand_sig)
        gaus.append(sig*np.random.randn(n//g, d)+np.random.rand(1,d))
    return np.concatenate(gaus)

def gauss_mix_long(n=10000,g=20,sigma=0.01,rand_sig=None, elongation=4):
    mean = [0, 0]
    cov = [[1, 0], [0, elongation]]*sigma  # diagonal covariance
    gaus=[]
    for _ in range(g):
        if rand_sig is None:
            sig = sigma
        else:
            sig = sigma*(np.random.rand()*rand_sig+1/rand_sig)
        x, y = np.random.default_rng().multivariate_normal(mean, cov*sig, n//g).T
        x=np.array(x).reshape(-1,1)
        y=np.array(y).reshape(-1,1)
        data=np.hstack((x,y))
        gaus.append(data)
    return np.concatenate(gaus)

#
# plot
#
def scat(P,s=1,c="r",ax=None):
    if ax is None:
        ax=plt
    if P is not None:
        ax.scatter(P[:,0],P[:,1],s=s,c=c)
        
def plot(X=None,C=None, frozen=None, title=None, E=None,img=True, ax=None):
    show = False
    if ax is None:
        show=True
        fig,ax = plt.subplots(1,1,figsize=(6,6))
            
    if img:
        c_size=15
        scat(X, s=1,c="g",ax=ax)
        if E is not None:
            E,U=get_error_and_util(X,C)
            E=E/E.sum()
            for i in range(E.shape[0]):
                scat(C[i:i+1], s = int(500*E[i]), c="limegreen", ax=ax)
        scat(frozen,225,c="cyan",ax=ax)
        scat(C,s=c_size,c="red",ax=ax)
        #ax.title="kokokok"
        ax.title.set_text(title)
        #plt.gca().set_aspect(1.0)
        #plt.show()
    else:
        print(title)
    if show:
        plt.show()
