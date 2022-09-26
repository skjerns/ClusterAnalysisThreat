# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:24:04 2022

Cluster analysis script for Threat-vs-Safety study

@author: Simon Kern
"""
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.stats import spatio_temporal_cluster_test

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


#%% data loading

data_dir = './Cluster_Threat-vs-Safe/'

conditions = ['Safe', 'Threat']

evoked = {cond:[] for cond in conditions}

montage = mne.channels.read_custom_montage('./AS-64_NO_REF.bvef')

for edf_file in os.listdir(os.path.abspath(data_dir)):
    raw = mne.io.read_raw(f'{data_dir}/{edf_file}')
    
    # need to ignore these, are not part of montage
    raw.drop_channels(['PO10', 'PO9'])
    
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

tmin = 0
tmax = 0.8

cluster_data = {}
for cond in conditions:
    data = [ev.get_data(tmin=tmin, tmax=tmax) for ev in evoked[cond]]
    cluster_data[cond] =  np.stack(data)
    
X = [np.transpose(x, (0, 2, 1)) for x in (cluster_data.values())]

adjacency, _ = mne.channels.find_ch_adjacency(raw.info, 'eeg')
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(X, 
                                                               adjacency=adjacency, 
                                                               tail=0, 
                                                               n_permutations=1000,
                                                               threshold=1.96,
                                                               n_jobs=-1)

grand_cropped = grand_diff.crop(tmin, tmax)

significant_points = np.zeros(grand_cropped.data.shape, dtype=bool)
for _cluster in clusters:
    for x, y in zip(*_cluster):
        significant_points[y, x] = True
        
fig_diff = grand_cropped.plot_joint(times = np.linspace(tmin, tmax, 7), title=f'Difference {" - ".join(conditions)}')

fig, axs = plt.subplots(1, 2); axs=axs.flatten()
fig_clust = grand_cropped.plot_image(show_names='all', axes=axs[0])
fig_clust = grand_cropped.plot_image(mask=significant_points, show_names='all', axes=axs[1])
