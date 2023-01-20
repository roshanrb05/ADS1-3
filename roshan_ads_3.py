import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize as opt
from sklearn import cluster 
import err_ranges as err

#defining file read function
def readfile(file_name):
    """
        Parameters
    ----------
    file_name : string
        full address of the file to be read.

    Returns
    -------
    d : dataframe
        input data as dataframe.
    d_trans : dataframe
        Transpose of the input dataframe.

    """
    #reading the file
    d = pd.read_excel(file_name)
    
    #removing unwanted columns
    d = d.drop(['Series Name', 'Series Code','Country Code'], axis = 1)
    
    #taking transpose
    d_trans = d.transpose()
    d_trans = d_trans.iloc[1:31,:]
    d_trans = d_trans.reset_index()
    d_trans = d_trans.rename(columns = {"index":"years", 0:"Arg", 1:"Bra"})
    
    d_trans = d_trans.dropna()
    #Renaming the header for transposed dataframe
    print(d_trans)
    
    #Cleaning the data
    d_trans["years"] = d_trans["years"].str[:4]
    d_trans["years"] = pd.to_numeric(d_trans["years"])
    d_trans["Arg"] = pd.to_numeric(d_trans["Arg"])
    d_trans["Bra"] = pd.to_numeric(d_trans["Bra"])
    print(d_trans)
    print(d_trans.dtypes)
    
    return d, d_trans
arab, arab_trans = readfile("C:\\Users\\babur\\Desktop\\Assignment 3\\arable_land.xlsx")
green, green_trans = readfile("C:\\Users\\babur\\Desktop\\Assignment 3\\greenhouse_emission.xlsx")

print(arab_trans)

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f   
green_trans["Narg"] = green_trans["Arg"]/green_trans["Arg"].abs().max()
param,cv = opt.curve_fit(exponential,green_trans["years"],green_trans["Narg"],p0=[4e8,0.02])
green_trans["fit"] = exponential(green_trans["years"],*param)
plt.figure()
plt.title("ARGENTINA")
plt.plot(green_trans["years"],green_trans["Narg"],label="data")
plt.plot(green_trans["years"],green_trans["fit"],label="fit")
sigma = np.sqrt(np.diag(cv))
low,up = err.err_ranges(green_trans["years"],exponential,param,sigma)
plt.fill_between(green_trans["years"],low,up,alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.title("ARGENTINA")
plt.plot(green_trans["years"],green_trans["Narg"],label="data")
pred = np.arange(1990,2040)
pred_ = exponential(pred,*param)
plt.plot(pred,pred_,label="pred")
plt.legend()
plt.show()

green_trans["Nbra"] = green_trans["Bra"]/green_trans["Bra"].abs().max()
param,cv = opt.curve_fit(exponential,green_trans["years"],green_trans["Nbra"],p0=[4e8,0.02])
green_trans["fit"] = exponential(green_trans["years"],*param)
plt.figure()
plt.title("BRAZIL")
plt.plot(green_trans["years"],green_trans["Nbra"],label="data")
plt.plot(green_trans["years"],green_trans["fit"],label="fit")
sigma = np.sqrt(np.diag(cv))
low,up = err.err_ranges(green_trans["years"],exponential,param,sigma)
plt.fill_between(green_trans["years"],low,up,alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.title("BRAZIL")
plt.plot(green_trans["years"],green_trans["Nbra"],label="data")
pred = np.arange(1990,2040)
pred_ = exponential(pred,*param)
plt.plot(pred,pred_,label="pred")
plt.legend()
plt.show()


Brazil = pd.DataFrame()
Brazil["arab"] = arab_trans["Bra"]
Brazil["green"] = green_trans["Bra"]

 
km = cluster.KMeans(n_clusters=2).fit(Brazil)
label = km.labels_
plt.figure()
plt.title("Brazil")
plt.scatter(Brazil["arab"],Brazil["green"],c=label,cmap="jet")
plt.xlabel("arable")
plt.ylabel("green")
c = km.cluster_centers_

for s in range(2):
    xc,yc = c[s,:]
    plt.plot(xc,yc,"dk",markersize=15)


arg = pd.DataFrame()
arg["arab"] = arab_trans["Arg"]
arg["green"] = green_trans["Arg"]

 
km = cluster.KMeans(n_clusters=2).fit(arg)
label = km.labels_
plt.figure()

          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          ")
plt.scatter(arg["arab"],arg["green"],c=label,cmap="jet")
plt.xlabel("arable")
plt.ylabel("green")
c = km.cluster_centers_
for s in range(2):
    xc,yc = c[s,:]
    plt.plot(xc,yc,"dk",markersize=15)
