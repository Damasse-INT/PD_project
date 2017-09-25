#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
__author__ = ["JB Damasse INT - CNRS"]

import os
import re
import glob
import scipy.io
import numpy as np
import _pickle as pkl
from scipy import stats as sp
import matplotlib.pyplot as plt
from lmfit.models import StepModel

def FOI_builder():
    '''
        This function is the File Of Interest builder, it extracts data from .mat arrays
        and experiment journal files
    '''
    participants = {}
    datas = {}
    for (c_path, c) in zip(['controls', 'IGT_like'], ['ctl', 'exp']):
        datas[c] = {}
        participants[c] = {}
        for cond in ['Healthy', 'On', 'Off']:
            participants[c][cond] = np.array([])
            datas[c][cond] = {}
            datapath = '../Data/' + c_path + '/' + cond + '/mat_structures/FINAL/'
            for infile in glob.glob(os.path.join(datapath, '*.mat')):
                DATA_MAT = scipy.io.loadmat(os.path.join(infile))
                s = int(re.search(r'\d+', infile).group())
                participants[c][cond] = np.concatenate((participants[c][cond], [int(s)]), axis=0)
                datas[c][cond][s] = {}
                for crit, types, code in zip(['position', 'velocity'], ['X', 'DX_filt'], [1, 9]): #1, 9 for X #2, 13 for Y
                    datas[c][cond][s][crit] = np.full((len(DATA_MAT['RES_STRUCT']['error'][0, 0]), 3000), np.nan)
                    for t in np.arange(len(DATA_MAT['RES_STRUCT']['error'][0, 0])):
                        length = DATA_MAT['RES_STRUCT']['trials'][0, 0][0, t][code].size
                        datas[c][cond][s][crit][t][0: length] = DATA_MAT['RES_STRUCT']['trials'][0, 0][0, t][code].flatten()

    jal = {}
    for (c_path, c) in zip(['controls/', 'IGT_like/'], ['ctl', 'exp']):
        jal[c] = {}
        for cond in ['Healthy', 'On', 'Off']:
            datapath = '../Data/' + c_path + cond + '/'
            jal[c][cond] = {}
            for infile in glob.glob(os.path.join(datapath, '*jal.mat')):
                jal_mat = scipy.io.loadmat(os.path.join(infile))
                s = int(re.search(r'\d+', infile).group())
                jal[c][cond][s] = np.full((jal_mat['jal']['data'][0][0].shape[0],
                                    jal_mat['jal']['data'][0][0].shape[1]), np.nan, dtype=np.dtype('object'))
                jal[c][cond][s] = jal_mat['jal']['data'][0][0]
    return(datas, jal, participants)

def traces_cleaner(datas, participants, jals, time_span=np.arange(0, 1400), types='velocity',
                    tolerance_crit=[65, 65, 65], Out_crit=1):
    '''
        This function builds on organized dict of all datas classifying them by experimental condition,
        Patients type, switch mode, and side.

        parameters:

        time_span = the time line of a trial as encoded by .mat struct file
        types = if you want to work on velocity or position
        tolerance_crit = Side type if chosen if the trace follow the tolerance criterion of func. I.E,
            by default if a trace has 65 %  of positive values, it will be considered rightward
        out_crit = the number of std tyou want to choose for outlying datas
    '''
    Cust_traces = {}
    for c in ['ctl', 'exp']:
        Cust_traces[c] = {}
        for cond, tol_percentage in zip(['Healthy', 'On', 'Off'], tolerance_crit):
            Cust_traces[c][cond] = {}
            for s in participants[c][cond]:
                temp_traces = datas[c][cond][s][types][:, time_span]
                Cust_traces[c][cond][s] = {}
                for sw in ['NS', 'PS', 'AS']:
                    Cust_traces[c][cond][s][sw] = {}
                    for side in ['right', 'left', 'other']:
                        Cust_traces[c][cond][s][sw][side] = np.full((datas[c][cond][s][types].shape[0],
                                                                      time_span.size), np.nan)
                if c == 'ctl':
                    sw = 'NS'
                    for idx in np.arange(temp_traces.shape[0]):
                        mask = ~np.isnan(temp_traces[idx])
                        if temp_traces[idx][mask].size == 0:
                            continue
                        else:
                            P_age_neg = (temp_traces[idx][mask][temp_traces[idx][mask] <= 0].size/
                                        temp_traces[idx][mask][~np.isnan(temp_traces[idx][mask])].size)*100
                            P_age_pos = (temp_traces[idx][mask][temp_traces[idx][mask] >= 0].size/
                                        temp_traces[idx][mask][~np.isnan(temp_traces[idx][mask])].size)*100
                            if P_age_pos > tol_percentage:
                                Cust_traces[c][cond][s][sw]['right'][idx] = temp_traces[idx]
                            elif P_age_neg > tol_percentage:
                                Cust_traces[c][cond][s][sw]['left'][idx] = temp_traces[idx]
                            else:
                                Cust_traces[c][cond][s][sw]['other'][idx] = temp_traces[idx]
                elif c == 'exp':
                    switch = np.where(jals[c][cond][s][:, 1] == jals[c][cond][s][:, 1][-1])[0][0]
                    for idx in np.arange(temp_traces.shape[0]):
                        if idx < switch:
                            sw = 'PS'
                        else:
                            sw = 'AS'
                        mask = ~np.isnan(temp_traces[idx])
                        if temp_traces[idx][mask].size == 0:
                            continue
                        else:
                            P_age_neg = (temp_traces[idx][mask][temp_traces[idx][mask] <= 0].size/
                                        temp_traces[idx][mask][~np.isnan(temp_traces[idx][mask])].size)*100
                            P_age_pos = (temp_traces[idx][mask][temp_traces[idx][mask] >= 0].size/
                                        temp_traces[idx][mask][~np.isnan(temp_traces[idx][mask])].size)*100
                            if P_age_pos > tol_percentage:
                                Cust_traces[c][cond][s][sw]['right'][idx] = temp_traces[idx]
                            elif P_age_neg > tol_percentage:
                                Cust_traces[c][cond][s][sw]['left'][idx] = temp_traces[idx]
                            else:
                                Cust_traces[c][cond][s][sw]['other'][idx] = temp_traces[idx]
                for sw in ['NS', 'PS', 'AS']:
                    for side in ['right', 'left', 'other']:
                        mask = np.all(np.isnan(Cust_traces[c][cond][s][sw][side]), axis=1)
                        Cust_traces[c][cond][s][sw][side] = Cust_traces[c][cond][s][sw][side][~mask]
                        error_vec = np.nanstd(Cust_traces[c][cond][s][sw][side], axis=0)
                        mean_vec = np.nanmean(Cust_traces[c][cond][s][sw][side], axis=0)
                        row_outliers, cols_outliers = np.where(
                            abs(Cust_traces[c][cond][s][sw][side]) > (abs(mean_vec)+(Out_crit*error_vec)))
                        for i, j in enumerate(row_outliers):
                            Cust_traces[c][cond][s][sw][side][j, cols_outliers[i]] = np.nan
    return Cust_traces

def traces_plotter(traces, participants, c, conditions, adj_axis=300, close_policy=False, fig_width=12, Out_crit=1):
    """
        Do i need a solid description for this ? probably be obsolete soon
    """
    for cond in conditions:
        if cond != 'Healthy':
            subjs = participants[c][cond][participants[c][cond] != 2]
        else:
            subjs = participants[c][cond]
        for s in subjs:
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.6180))
            for side, col, l_style in zip(['left', 'right'], ['r', 'g'], ['-', '--']):
                if c == 'ctl':
                    sw_list = ['NS']
                elif c == 'exp':
                    sw_list = ['PS', 'AS']
                for sw in sw_list:
                    for i in np.arange(traces[c][cond][s][sw][side].shape[0]):
                        X = np.arange(traces[c][cond][s][sw][side][i].size)-adj_axis
                        Y = traces[c][cond][s][sw][side][i]
                        _ = ax.plot(X, Y, color=col)

                    _ = ax.plot(X, np.nanmean(traces[c][cond][s][sw][side], axis=0), color='k', linestyle=l_style,
                                linewidth=3, label='mean {} side velocity'.format(side))
                    error_vec = Out_crit*np.nanstd(traces[c][cond][s][sw][side], axis=0)
                    _ = ax.fill_between(X, np.nanmean(traces[c][cond][s][sw][side], axis=0)-error_vec,
                            np.nanmean(traces[c][cond][s][sw][side], axis=0)+error_vec, facecolor=col, alpha=0.5)
                ax.set_ylabel('Smooth eye velocity (°/s)', fontsize=14)
                ax.set_xlabel('Time', fontsize=11)
                ax.set_xlim([-100, 1000])
                ax.set_ylim([-10, 10])
                _ = ax.set_title('raw {} traces for S = {} N°{}'.format(c, cond, int(s)))
                _ = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        if close_policy:
            plt.close('all')

def vec_latency_unc(traces, participants, conditions, c, sw='NS', adj_axis=300,
        crit = 0.5, Out_crit=1.5, close_policy=False, fig_width=12):

    def nan_helper(y):
        """
            Helper to handle indices and logical indices of NaNs.
            Main reason of that code ? => Avoid errors when using lmfit
            see use below for an example
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    if c == 'ctl':
        for cond in conditions:
            # excluded participants
            if cond != 'Healthy':
                subjs = participants[c][cond][participants[c][cond] != 2]
            else:
                subjs = participants[c][cond]
            for s in subjs:
                fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.6180))
                latencies = []
                for side, col, l_style in zip(['left', 'right'], ['turquoise', 'b'], ['-', '--']):
                    Y = np.nanmean(traces[c][cond][s][sw][side], axis=0)
                    X = np.arange(Y.size)-adj_axis
                    # Y interpolation of NaNs
                    nans, x= nan_helper(Y)
                    Y[nans] = np.interp(x(nans), x(~nans), Y[~nans])
                    _ = ax.plot(X, Y, color=col, linestyle=l_style,
                                    linewidth=3, label='mean {} side velocity'.format(side))
                    error_vec = Out_crit*np.nanstd(traces[c][cond][s][sw][side], axis=0)
                    _ = ax.fill_between(X, Y-error_vec, Y+error_vec, facecolor=col, alpha=0.3)

                    # Mean trace smoothing using Levenberg–Marquardt algorithm
                    mod =  StepModel(form='erf')
                    pars = mod.guess(Y, x=X)
                    out = mod.fit(Y, pars, x=X)
                    ax.plot(np.asarray(X), out.best_fit, color='k', label= '{} trace LM model'.format(side))
                    full_size = traces[c][cond][s][sw]['left'].shape[0]+traces[c][cond][s][sw]['right'].shape[0]
                    perc_side = traces[c][cond][s][sw][side].shape[0]/full_size
                    list_l = []
                    for tps in range(len(X)) :
                        if perc_side*abs(out.best_fit[tps]) > crit:
                            list_l.append(X[tps])
                    if len(list_l)!=0:
                        latencies.append(list_l[0])
                    else:
                        ax.text(adj_axis, 1.5, "NO LATENCY FOUND !", color='r', fontsize=15)
                ax.axvline(np.mean(latencies), color='k', linewidth=3, label='latency = {} ms'.format(np.mean(latencies)))
                ax.set_ylabel('Smooth eye velocity (°/s)', fontsize=14)
                ax.set_xlabel('Time', fontsize=11)
                ax.set_xlim([-100, 1000])
                ax.set_ylim([-10, 10])
                _ = ax.set_title('raw {} traces for S = {} N°{}'.format(c, cond, int(s)))
                _ = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                if close_policy:
                    plt.close('all')
    elif c == 'exp':
        for cond in conditions:
            # excluded participants
            if cond != 'Healthy':
                subjs = participants[c][cond][participants[c][cond] != 2]
            else:
                subjs = participants[c][cond]
            for s in subjs:
                fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.6180))
                latencies = []
                for side, col, l_style in zip(['left', 'right'], ['r', 'g'], ['-', '--']):
                    Y = np.nanmean(np.concatenate((traces[c][cond][s]['PS'][side],
                            traces[c][cond][s]['AS'][side]), axis=0), axis=0)
                    X = np.arange(Y.size)-adj_axis

                    # Y interpolation of NaNs
                    nans, x= nan_helper(Y)
                    Y[nans] = np.interp(x(nans), x(~nans), Y[~nans])
                    _ = ax.plot(X, Y, color='k', linestyle=l_style,
                                    linewidth=3, label='mean {} side velocity'.format(side))
                    error_vec = Out_crit*np.nanstd(np.concatenate((traces[c][cond][s]['PS'][side],
                                    traces[c][cond][s]['AS'][side]), axis=0), axis=0)
                    _ = ax.fill_between(X, Y-error_vec, Y+error_vec, facecolor=col, alpha=0.5)

                    # Mean trace smoothing using Levenberg–Marquardt algorithm
                    mod =  StepModel(form='erf')
                    pars = mod.guess(Y, x=X)
                    out = mod.fit(Y, pars, x=X)
                    ax.plot(np.asarray(X), out.best_fit, color='k', label= '{} trace LM model'.format(side))
                    full_size = np.concatenate((traces[c][cond][s]['PS']['left'],
                           traces[c][cond][s]['AS']['left']), axis=0).shape[0]+np.concatenate(
                           (traces[c][cond][s]['PS']['right'], traces[c][cond][s]['AS']['right']), axis=0).shape[0]
                    perc_side = np.concatenate((traces[c][cond][s]['PS'][side],
                            traces[c][cond][s]['AS'][side]), axis=0).shape[0]/full_size
                    list_l = []
                    for tps in range(len(X)) :
                        if perc_side*abs(out.best_fit[tps]) > crit:
                            list_l.append(X[tps])
                    if len(list_l)!=0:
                        latencies.append(list_l[0])
                    else:
                        ax.text(adj_axis, 1.5, "NO LATENCY FOUND !", color='r', fontsize=15)
                ax.axvline(np.mean(latencies), color='k', linewidth=3, label='latency = {} ms'.format(np.mean(latencies)))
                ax.set_ylabel('Smooth eye velocity (°/s)', fontsize=14)
                ax.set_xlabel('Time', fontsize=11)
                ax.set_xlim([-100, 1000])
                ax.set_ylim([-10, 10])
                _ = ax.set_title('raw {} traces for S = {} N°{}'.format(c, cond, int(s)))
                _ = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                if close_policy:
                    plt.close('all')

def vec_latency_VX_vs_T(traces, participants, conditions, journal, sw_c = ['NS', 'NS'], sw_e = ['PS', 'AS'], adj_axis=300,
        crit = 0.01, Out_crit=1.5, close_policy=False, fig_width=12):

    def nan_helper(y):
        """
            Helper to handle indices and logical indices of NaNs.
            Main reason of that code ? => Avoid errors when using lmfit
            see use below for an example
        """
        return np.isnan(y), lambda z: z.nonzero()[0]

    for cond in conditions:
        # excluded participants
        if cond != 'Healthy':
            subjs = participants['ctl'][cond][participants['ctl'][cond] != 2]
        else:
            subjs = participants['ctl'][cond]
        for s in subjs:
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width/1.6180))
            for c, switches, col_code, VA_col in zip(['ctl', 'exp'], [sw_c, sw_e],
                                            [['turquoise', 'b'], ['r', 'g']], ['b', 'k']):
                latencies = []
                for side, col, l_style in zip(['left', 'right'], col_code, ['-', '--']):
                    Y = np.nanmean(np.concatenate((traces[c][cond][s][switches[0]][side],
                                       traces[c][cond][s][switches[1]][side]), axis=0), axis=0)
                    X = np.arange(Y.size)-adj_axis

                    # Y interpolation of NaNs
                    nans, x = nan_helper(Y)
                    Y[nans]= np.interp(x(nans), x(~nans), Y[~nans])
                    #_ = ax.plot(X, Y, color=col, linestyle=l_style,
                    #                 linewidth=3, label='mean {} side velocity'.format(side))
                    #error_vec = Out_crit*np.nanstd(np.concatenate((traces[c][cond][s][switches[0]][side],
                #                        traces[c][cond][s][switches[1]][side]), axis=0), axis=0)
                #    _ = ax.fill_between(X, Y-error_vec, Y+error_vec, facecolor=col, alpha=0.3)

                    # Mean trace smoothing using Levenberg–Marquardt algorithm
                    mod =  StepModel(form='erf')
                    pars = mod.guess(Y, x=X)
                    out = mod.fit(Y, pars, x=X)
                    ax.plot(np.asarray(X), out.best_fit, color=col)
                    current = np.concatenate((traces[c][cond][s][switches[0]][side],
                                       traces[c][cond][s][switches[1]][side]), axis=0).shape[0]
                    sum_of = np.concatenate((traces[c][cond][s][switches[0]]['left'],
                                       traces[c][cond][s][switches[1]]['left']), axis=0).shape[0] + np.concatenate((traces[c][cond][s][switches[0]]['right'],
                                               traces[c][cond][s][switches[1]]['right']), axis=0).shape[0]

                    perc_side = current/sum_of
                    list_l = []
                    for tps in range(len(X)) :
                        if perc_side*abs(out.best_fit[tps]) > crit:
                            list_l.append(X[tps])
                    if len(list_l)!=0:
                        latencies.append(list_l[0])
                    else:
                        ax.text(adj_axis, 1.5, "NO LATENCY FOUND !", color='r', fontsize=15)
                ax.axvline(np.mean(latencies), color=VA_col, linewidth=3,
                           label='latency {} = {} ms'.format(c, np.mean(latencies)))
                ax.set_ylabel('Smooth eye velocity (°/s)', fontsize=14)
                ax.set_xlabel('Time', fontsize=11)
                ax.set_xlim([-100, 1000])
                ax.set_ylim([-8, 8])
                _ = ax.set_title('{} {} dep rule = {}'.format(cond, int(s), journal[c][cond][s][:, 1][0][0]))
                _ = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
                if close_policy:
                    plt.close('all')
