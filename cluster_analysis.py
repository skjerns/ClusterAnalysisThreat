# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:24:04 2022

Cluster analysis script for Threat-vs-Safety study

@author: Simon Kern
"""
import os
import mne
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from mne.stats import spatio_temporal_cluster_1samp_test

#%% data loading

data_dir = './Cluster_Threat-vs-Safe/'
conditions = ['Safe', 'Threat']

evoked = {cond:[] for cond in conditions}
montage1 = mne.channels.read_custom_montage('./AC-64.bvef')
montage2 = mne.channels.read_custom_montage('./AP-64.bvef')
ch_pos = montage1.get_positions()['ch_pos'] | montage2.get_positions()['ch_pos']
montage = mne.channels.make_dig_montage(ch_pos=ch_pos)


for edf_file in os.listdir(os.path.abspath(data_dir)):
    
    raw = mne.io.read_raw(f'{data_dir}/{edf_file}') 
    raw.set_montage(montage)

    # infer which condition this file belongs to
    for cond in conditions:
        if cond in edf_file:
            break
    evoked[cond].append(mne.EvokedArray(raw.get_data(), raw.info))

# creating grand averages
grands = {cond: mne.grand_average(evoked[cond]) for cond in conditions}
grand_diff = mne.combine_evoked(list(grands.values()) , weights=[1, -1])



#%% cluster analysis
tmin = 0.1
tmax = 0.6

data = {}
for cond in conditions:
    data[cond] = [ev.get_data(tmin=tmin, tmax=tmax) for ev in evoked[cond]]

  
adjacency, _ = mne.channels.find_ch_adjacency(raw.info, 'eeg')

# np.save('adjacency.npy', adjacency)
adjacency = sparse.load_npz('adjacency.npz')

X = np.array(data['Threat']) - np.array(data['Safe'])
X = np.transpose(X, (0, 2, 1))
     
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(X, 
                                                                     adjacency=adjacency, 
                                                                     tail=1, 
                                                                     n_permutations=1000,
                                                                     out_type='mask',
                                                                     threshold=1.96,
                                                                     n_jobs=-1)

clusters = np.array(clusters)
grand_cropped = grand_diff.crop(tmin, tmax)

for i, (cluster, pval) in enumerate(zip(clusters, cluster_pv)):
    if pval>0.05:
        print(f'Cluster {i} not significant {pval=}')
        continue
    print(f'Cluster {i} significant {pval=}')
    cluster = np.array(cluster)
    significant_points = np.pad(cluster, [[1, 0], [0,0]]).T   
    fig = grand_cropped.plot_image(show_names='all', mask=significant_points)
    plt.title(f'Cluster {i}, {pval=}')
