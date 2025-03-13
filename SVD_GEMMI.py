#!/usr/bin/env python3
##################################################
#

# 
##################################################
import dask.array as da
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mp
import gemmi

# cd f'N:\Documents\articles\CracRy\SVD_maps191A_alldata_vs_truedark'

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


def convert_mtz_to_matrix(mtz_path):
    mtz=gemmi.read_mtz_file(mtz_path)
    gridsize=mtz.get_size_for_hkl()
    recigrid=mtz.get_f_phi_on_grid('FWT', 'PHWT', gridsize)
    grid=gemmi.transform_f_phi_grid_to_map(recigrid)
    nx, ny, nz = grid.nu, grid.nv, grid.nw
    numpy_array = np.empty((nx, ny, nz), dtype=np.float32)

    # Iterate over each point in the grid
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                numpy_array[x, y, z] = grid.get_value(x, y, z)

    return numpy_array



def write_matrix_to_mtz(matrix, mtz_input_path, mtz_output_path): #there is a trailing variable that makes it work, the function version does not work 
    mtz=gemmi.read_mtz_file(mtz_input_path)
    resolution_limit=mtz.resolution_high()
    gridsize=mtz.get_size_for_hkl()
    expgrid=mtz.get_f_phi_on_grid('FWT', 'PHWT', gridsize)
    rs_grid_template=gemmi.transform_f_phi_grid_to_map(expgrid)
    nx, ny, nz = rs_grid_template.nu, rs_grid_template.nv, rs_grid_template.nw
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                rs_grid_template.set_value(x, y, z,matrix[x, y, z])
    
    recigrid = gemmi.transform_map_to_f_phi(rs_grid_template, half_l=True)
    data = recigrid.prepare_asu_data(dmin=resolution_limit)
    
    mtz.set_data(data)
    mtz.write_to_file(mtz_output_path)








os.chdir('./')
directory = "./"  
map_list={}     
files=[]
timescales=[]
for entry in os.scandir(directory):   
    if entry.path.endswith("_trimmed.mtz") and entry.is_file() and entry.path.startswith("./diff_"):   
        a=entry.path[2:-4]                               
        print(entry.path[7:-12])
        files.append(a)
        tim=int(entry.path[7:-14])
        
        if 'us_trimmed.mtz' in entry.path:
            tim*=1000
        if 'ms_trimmed.mtz' in entry.path:
            tim*=1000000
        timescales.append(tim)
        print(tim,a)



ordered_files=pd.DataFrame(data=files,index=timescales).sort_index()

n = len(list(ordered_files[0]))

try:
    os.mkdir('output')
    print("Directory " , 'output' ,  " Created ")
except FileExistsError:
    print("Directory " , 'output' ,  " already exists")  


#%% test that they have the same dimensions and perform SVD

shape_map=[]
dimaps=[]
for i in ordered_files[0]:
    vars()["map_"+str(i)]= convert_mtz_to_matrix(i+'.mtz')
    shape_map.append(vars()["map_"+str(i)].shape)




m = np.prod(vars()["map_"+str(i)].data.shape)   
A = np.zeros((m,n),dtype=np.float32)   
rank=0
for i in ordered_files[0]: 
    A[:,rank] = vars()['map_'+str(i)].reshape(-1)
    rank+=1

B=A

A=da.array(B)

U,S,VT=da.linalg.svd(A)
U=np.array(U)
S=np.array(S)
VT=np.array(VT)


#%% exporting maps
#%%% raw structural elements
# rank=0
for rank in range(n):
    write_matrix_to_mtz(U[:,rank].reshape(vars()['map_'+list(ordered_files[0])[0]].shape), list(ordered_files[0])[0]+'.mtz', 'output/lSV_'+str(rank)+'.mtz')
    # rank+=1



#%%% scaled (by their SV) structural elements
# sigemmiatrix=np.array(np.zeros(shape=(m,n)),dtype=np.float32)
sigemmiatrix=np.array(np.zeros((n,n),dtype=np.float32))
for i in range(0,len(S)):
    print(i,S[i])
    sigemmiatrix[i,i]=S[i]


US=np.array(np.matrix(U) @ np.matrix(sigemmiatrix))

for rank in range(n):
    write_matrix_to_mtz(US[:,rank].reshape(vars()['map_'+list(ordered_files[0])[0]].shape), list(ordered_files[0])[0]+'.mtz', 'output/SVscaled_lSV_'+str(rank)+'.mtz')
    # rank+=1


#%%% test recomposed map 1

Ar1=np.array(US @ VT.T[0,:])

write_matrix_to_mtz(Ar1.reshape(vars()['map_'+list(ordered_files[0])[0]].shape), list(ordered_files[0])[0]+'.mtz', 'output/reformed_'+list(ordered_files[0])[0]+'.mtz')



#%% plot magnitude over time 
scaled_time_factors=(sigemmiatrix @ VT)[0:n,:]



tmp={'time_ns':list(ordered_files.index)}

plt.rcParams["figure.figsize"] = (15/2.54,7.5/2.54)
fig, ax = plt.subplots()     
ax.set_xlabel('time-point (ms)', fontsize=10)  
ax.xaxis.set_label_coords(x=0.5, y=-0.08)      
ax.set_ylabel('scaled rSV', fontsize=10)               
ax.yaxis.set_label_coords(x=-0.04, y=0.5)       
palette=sns.color_palette(palette='bright', n_colors=2)   
for i in range(0,2):
    tmp['SVn'+str(i)]=scaled_time_factors[i]
    ax.plot(np.array(ordered_files.index),scaled_time_factors[i], 
            linewidth=2, 
            linestyle='dashed',
            marker='o',    
            markersize=10,  
            alpha=0.5,             
            label='SV n° ' + str(i) ,
            color=palette[i])               

# plt.xscale("log")

ax.xaxis.set_ticks(list(ordered_files.index))
test=ax.xaxis.get_ticklabels()
newlabel=[]
for i in ordered_files.index:
    newlabel.append(mp.text.Text(int(i), 0, re.sub('_','',re.sub('o','',re.sub('F','',re.sub('d','',re.sub('3_3','3.3',ordered_files[0][i][5:-8])))))))
ax.set_xticklabels(newlabel)
ax.set_title('rSV0 and rSV1 over time, scaled by their SV', fontsize=10, fontweight='bold')  
ax.set_xlim([10,233000000])
ax.tick_params(labelsize=8)

legend = plt.legend(loc='best', shadow=True, prop={'size':4})


figfilename = "Cter-restrained_SVD_CraCRY_SV-scaled_rSV.pdf"             
plt.savefig(figfilename, dpi=900, transparent=True,bbox_inches='tight') 
figfilename = "Cter-restrained_SVD_CraCRY_SV-scaled_rSV.png"             
plt.savefig(figfilename, dpi=900, transparent=True,bbox_inches='tight') 
figfilename = "Cter-restrained_SVD_CraCRY_SV-scaled_rSV.svg"             
plt.savefig(figfilename, dpi=900, transparent=True,bbox_inches='tight') 
plt.close()

pd.DataFrame(tmp, index=tmp['time_ns']).to_csv("Cter-restrained_SVD_CraCRY_SV-scaled_rSV.csv", index=False)




#%% closing maps 














