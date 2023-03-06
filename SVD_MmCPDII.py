#!/usr/bin/env python3
##################################################
#

# 
##################################################
import shutil
import mrcfile

import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mp


#%% read maps
def all_equal(iterator):
    """
    Check if all elements in an iterator are equal
    Parameters:
    - iterator: an iterator containing elements of any type
    Returns:
    - bool: True if all elements are equal, False otherwise
    """
    return len(set(iterator)) <= 1

def collapsdim(iterator):
    """
    Collapse the dimensions of a given iterator
    Parameters:
    - iterator: an iterator containing sublists of length 3
    Returns:
    - tuple: containing the collapsed dimensions of the iterator, the index of the minimum dimension and the index of the maximum dimension 
    """
    dimaps=[]
    nmin=0
    nmax=0
    for i in range(0,len(iterator)):
        dimaps.append(0)
        for j in [0,1,2]:
            dimaps[i]+=iterator[i][j]
        if dimaps[nmin]>dimaps[i]:
            nmin=i
        if dimaps[nmax]<dimaps[i]:
            nmax=i
    return(dimaps,nmin,nmax)



os.chdir('./')
directory = "./"  
map_list={}     
files=[]
timescales=[]
for entry in os.scandir(directory):   
    if entry.path.endswith(".map") and entry.is_file() and "restricted" in entry.path and "reshaped" not in entry.path:   
        a=entry.path[2:-4]                               
        
        files.append(a)
        tim=float(re.sub('_', '.', entry.path[19:-6]))
        
        if 'ns.map' in entry.path:
            tim*=1000
        if 'us.map' in entry.path:
            tim*=1000000
        timescales.append(int(tim))
        print(tim,a)



ordered_files=pd.DataFrame(data=files,index=timescales).sort_index()




n = len(list(ordered_files[0]))
mx = mrcfile.open(str(list(ordered_files[0])[0])+'.map').header.mx
my = mrcfile.open(str(list(ordered_files[0])[0])+'.map').header.my
mz = mrcfile.open(str(list(ordered_files[0])[0])+'.map').header.mz
try:
    os.mkdir('output')
    print("Directory " , 'output' ,  " Created ")
except FileExistsError:
    print("Directory " , 'output' ,  " already exists")  


#%% test that they have the same dimensions and perform SVD

shape_map=[]
dimaps=[]
for i in range(n):
    vars()["mrc"+str(i)]= mrcfile.open(str(list(ordered_files[0])[i]+'.map'),mode='r')
    shape_map.append(vars()["mrc"+str(i)].data.shape)




if all_equal(shape_map):
    m = np.prod(mrc0.data.shape)   
    A = np.zeros((m,n),dtype=np.float32)   
else : 
    print('Unequal map dimensions, trimming all maps to fit the smallest')
    mapdim, rankmin, rankmax=collapsdim(shape_map)
    temp_map_store=np.array(vars()["mrc"+str(rankmax)].data) 
    temp_dest_store=np.array(vars()["mrc"+str(rankmin)].data) 
    for x in range(0,len(temp_dest_store)):
        for y in range(0,len(temp_dest_store[0,:])):
            for z in range(0,len(temp_dest_store[0,0])):
                temp_dest_store[x,y][z]=temp_map_store[x,y][z]
    shutil.copy(str(list(ordered_files[0])[rankmin])+'.map',str('reshaped_'+str(list(ordered_files[0])[0])+'.map'))
    mrctest=mrcfile.open(str('reshaped_'+str(list(ordered_files[0])[0])+'.map'),mode='r+')
    mrctest.set_data(temp_dest_store)
    mrctest.header.mx=mx
    mrctest.header.my=my
    mrctest.header.mz=mz
    mrctest.close()
for i in range(n): 
    A[:,i] = locals()['mrc'+str(i)].data.reshape(-1) 



U, S, VT = np.linalg.svd(A) 

#%% exporting maps
#%%% raw structural elements
for i in range(0,n):
    shutil.copy(str(list(ordered_files[0])[0])+'.map','output/svd_'+str(i)+'_SV.map')
    vars()["svd"+str(i)]=mrcfile.open('output/svd_'+str(i)+'_SV.map',mode='r+')
    vars()["svd"+str(i)].set_data(U[:,i].reshape(vars()["svd"+str(i)].data.shape))
    vars()["svd"+str(i)].header.mx = mx
    vars()["svd"+str(i)].header.my = my
    vars()["svd"+str(i)].header.mz = mz
    vars()["svd"+str(i)].close()



#%%% scaled (by their SV) structural elements
sigmatrix=np.array(np.zeros(shape=(m,n)),dtype=np.float32)

for i in range(0,len(S)):
    print(i,S[i])
    sigmatrix[i,i]=S[i]


US=np.matrix(U) @ np.matrix(sigmatrix)

for i in range(0,n):
    shutil.copy(str(list(ordered_files[0])[0])+'.map','output/scaled_svd_'+str(i)+'_SV.map')
    vars()["scaled_svd"+str(i)]=mrcfile.open('output/scaled_svd_'+str(i)+'_SV.map',mode='r+')
    vars()["scaled_svd"+str(i)].set_data(U[:,i].reshape(vars()["scaled_svd"+str(i)].data.shape))
    vars()["scaled_svd"+str(i)].header.mx = mx
    vars()["scaled_svd"+str(i)].header.my = my
    vars()["scaled_svd"+str(i)].header.mz = mz
    vars()["scaled_svd"+str(i)].close()


#%%% test recomposed map 1

Ar1=np.array(US @ VT.T[0,:])



shutil.copy(str(list(ordered_files[0])[1])+'.map','output/reformed_map0.map')
vars()['reformed_map0']=mrcfile.open('output/reformed_map0.map',mode='r+')
vars()['reformed_map0'].set_data(Ar1.reshape(vars()['reformed_map0'].data.shape))
vars()['reformed_map0'].header.mx = mx
vars()['reformed_map0'].header.my = my
vars()['reformed_map0'].header.mz = mz
vars()['reformed_map0'].close()


#%% plot magnitude over time 
scaled_time_factors=(sigmatrix @ VT)[0:n,:]



tmp={'time_ps':list(ordered_files.index)}

plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
fig, ax = plt.subplots()     
ax.set_xlabel('time-point (s, in log scale)', fontsize=10)  
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
ax.set_ylabel('Magnitude', fontsize=10)               
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
palette=sns.color_palette(palette='Spectral', n_colors=len(S))   
for i in range(0,len(S)):
    tmp['SVn'+str(i)]=scaled_time_factors[i]
    ax.plot(np.array(ordered_files.index),scaled_time_factors[i], 
            linewidth=1,                    
            label='SV n° ' + str(i) ,
            color=palette[i])               



plt.xscale("log")


ax.xaxis.set_ticks(list(ordered_files.index))
test=ax.xaxis.get_ticklabels()
newlabel=[]
for i in ordered_files.index:
    newlabel.append(mp.text.Text(int(i), 0, re.sub('_','',re.sub('o','',re.sub('F','',re.sub('d','',re.sub('3_3','3.3',ordered_files[0][i][-8:])))))))
ax.set_xticklabels(newlabel)
ax.set_title('Magnitude of each of the structural components over the maps', fontsize=10, fontweight='bold')  

ax.tick_params(labelsize=8)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})


figfilename = "magnitude_over_time_log.pdf"             
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "magnitude_over_time_log.png"             
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "magnitude_over_time_log.svg"             
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()

pd.DataFrame(tmp, index=tmp['time_ps']).to_csv("Magnitude_overtime_ps_5A_CPD_SwissFEL.csv", index=False)





plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
fig, ax = plt.subplots()     
ax.set_xlabel('time-point (s, linear scale)', fontsize=10)  
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
ax.set_ylabel('Weight', fontsize=10)               
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
palette=sns.color_palette(palette='Spectral', n_colors=n)   
for i in range(0,n):
    ax.plot(np.array(ordered_files.index),scaled_time_factors[i], 
            linewidth=1,                    
            label='SV n° ' + str(i) ,
            color=palette[i])               






ax.xaxis.set_ticks(list(ordered_files.index))
test=ax.xaxis.get_ticklabels()
newlabel=[]
for i in ordered_files.index:
    newlabel.append(mp.text.Text(int(i), 0, re.sub('_','',re.sub('o','',ordered_files[0][i][-5:]))))
ax.set_xticklabels(newlabel)
ax.set_title('Magnitude of the structural components over the maps', fontsize=10, fontweight='bold')  

ax.tick_params(labelsize=8)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})





figfilename = "SV_magnitude_over_time_lin.pdf"             
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "SV_magnitude_over_time_lin.png"             
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
figfilename = "SV_magnitude_over_time_lin.svg"             
plt.savefig(figfilename, dpi=300, transparent=True,bbox_inches='tight') 
plt.close()










poids=np.abs(scaled_time_factors)
for i in range(0,len(poids)):
    print(poids[i])
    totpoids=sum(poids[:,i])
    for j in range(0,len(poids[:,i])):
        poids[j][i]/=totpoids



tmp={'time_ps':list(ordered_files.index)}

plt.rcParams["figure.figsize"] = (20/2.54,15/2.54)
fig, ax = plt.subplots()     
ax.set_xlabel('time-point (s, linear scale)', fontsize=10)  
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
ax.set_ylabel('Weight ', fontsize=10)               
ax.yaxis.set_label_coords(x=-0.1, y=0.5)       
palette=sns.color_palette(palette='Spectral', n_colors=n)   
for i in range(0,n):
    tmp['SVn'+str(i)]=scaled_time_factors[i]
    ax.plot(np.array(ordered_files.index),poids[i], 
            linewidth=1,                    
            label='SV n° ' + str(i) ,
            color=palette[i])               





plt.xscale('log')        
ax.xaxis.set_ticks(list(ordered_files.index))
test=ax.xaxis.get_ticklabels()
newlabel=[]
for i in ordered_files.index:
    newlabel.append(mp.text.Text(int(i), 0, ordered_files[0][i][17:]))
ax.set_xticklabels(newlabel)
ax.set_title('weight of each of the structural components over the maps', fontsize=10, fontweight='bold')  

ax.tick_params(labelsize=8)

legend = plt.legend(loc='upper right', shadow=True, prop={'size':8})


    


figfilename = "SV_weight_over_time_log.pdf"             
plt.savefig(figfilename, dpi=900, transparent=True,bbox_inches='tight') 
figfilename = "SV_weight_over_time_log.png"             
plt.savefig(figfilename, dpi=900, transparent=True,bbox_inches='tight') 
figfilename = "SV_weight_over_time_log.svg"             
plt.savefig(figfilename, dpi=900, transparent=True,bbox_inches='tight') 
plt.close()

pd.DataFrame(tmp, index=tmp['time_ps']).to_csv("Weights_overtime_ps_5A_CPD_SwissFEL.csv", index=False)



#%% closing maps 

for i in range(n):
    vars()["mrc"+str(i)].close()
print("SVD on maps performed successfully!")















