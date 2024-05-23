'''
This file is mainly about importing data and preprocessing it through nmrpy; for the base functions, it is recommended that you check out their repo
https://github.com/NMRPy/nmrpy
'''

import nmrpy
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import time
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib
from matplotlib import colormaps
'''
Please read the maine page or the readme for an explanation on the directory formatting requirements. Alternatively, check out the ipynb for examples
This file is for processing files with arrays as checkpoints
'''

def import_data(file_dir, key):    
    init_arr = os.listdir(f"./{file_dir}/")

    exp_names = []
    for i in range(len(init_arr)):
        if("." not in init_arr[i]):
            exp_names.append(init_arr[i])

    fidarr_dict = {}
    for j in range(len(exp_names)):
        file_arr = os.listdir(f"./{file_dir}/{exp_names[j]}/{key}/")
        temp_dict = {}
        for k in range(len(file_arr)):
            if(file_arr[k] != '.DS_Store'):
                temp_dict[k] = nmrpy.from_path(fid_path=f"./{file_dir}/{exp_names[j]}/{key}/{file_arr[k]}")
        fidarr_dict[f"{exp_names[j]}"] = temp_dict

    return fidarr_dict

def preprocess(fid_array: nmrpy.data_objects.FidArray):
    start_time = time.time()
    print("EMHZ, Fourier Transform")
    em_ft(fid_array)
    print("Finished!\nPhase Correct, Real, Norm Fids:\n")
    pc(fid_array)
    print("Finished!")
    print("-- %5.5f s Run Time --" % (time.time() - start_time))

def em_ft(fid_array: nmrpy.data_objects.FidArray):
    fid_array.emhz_fids();
    fid_array.ft_fids();

def pc(fid_array: nmrpy.data_objects.FidArray):
    fid_array.phase_correct_fids(verbose = False);
    fid_array.real_fids();
    fid_array.norm_fids();

def generate_idx(n, arr):
    m = len(arr)
    out = []
    for i in range(n):
        n = str(np.random.randint(0,m))
        while(n in out):
            n = str(np.random.randint(0,m))
        s = str(n)
        while(len(s) != len(str(m))):
            s = "0" + s
        out.append(s)
    return out

def avg_spectra(fidArray: nmrpy.data_objects.FidArray, n):
    dataOut = 0
    ppmOut = 0
    indx_array = generate_idx(n, fidArray.data)
    #print(indx_array)
    for i in range(n):
        dataOut = dataOut + fidArray.get_fid(f"fid{indx_array[i]}").data
        ppmOut = ppmOut + fidArray.get_fid(f"fid{indx_array[i]}")._ppm
    dataOut = dataOut / n
    ppmOut = ppmOut / n
    return dataOut, ppmOut

def preproc_max(dataDict):
    max_avgs = []
    ppms = []
    for i in dataDict.keys():
        for j in dataDict[i].keys():
            preprocess(dataDict[i][j])
            tTuple = avg_spectra(dataDict[i][j], len(dataDict[i][j].data))
            max_avgs.append(tTuple[0])
            ppms.append(tTuple[1])
    return max_avgs, ppms

def plotall(sList, ppm, cmap = 'viridis', label = "sample"):
    if(len(sList) != len(ppm)):
       print("Lengths are different, plot failed")
    else:
        map = matplotlib.colormaps[cmap].resampled(len(ppm))
        for i in range(len(ppm)):
            plt.plot(ppm[i], sList[i], c = map.colors[i], label = f"{label}_{i}")

def shift_tms(fid_arr, ppm):
    pos = np.where(fid_arr == max(fid_arr))
    zero = np.where((ppm == min(abs(ppm))) | (ppm == -1 * min(abs(ppm))))
    shift = True
    if(pos != zero):
        print(f"\tPosition Not Equal: \tfidpos: {pos[0][0]}\tppmpos: {zero[0][0]}, \tppm value: {ppm[zero[0][0]]}\n")
    else:
        print(f"\tPosition Equal: \tfidpos: {pos[0][0]}\tppmpos: {zero[0][0]}, \tppm value: {ppm[zero[0][0]]}\n")
        shift = False
        newFID = fid_arr

    if shift:
        print("\tShifting fid such that the highest frequency peak is also closest to 0\n")
        diff = zero[0][0] - pos[0][0]
        print(f"\tShift amount = {diff}\n")
        if (diff > (len(fid_arr)/10)):
            print("\tToo much shift. Alignment Failed\n")
            return "emp","ty"
        else:
            newFID = np.roll(fid_arr, diff)
            shift_tms(newFID, ppm)

    #set min ppm to 0
    
    ppmval = ppm[zero]

    new_ppm = ppm - ppmval

    pos = np.where(newFID == max(newFID))
    zero = np.where((new_ppm == min(abs(new_ppm))) | (new_ppm == -1 * min(abs(new_ppm))))
        
    print(f"final values: \tfidpos: {pos[0][0]}\tppmpos: {zero[0][0]}, \tppm value: {new_ppm[zero[0][0]]}\n")
    
    return newFID, new_ppm

def rem_invalid_shifts(shift, ppmSh):
    shift = [j for j in shift if type(j) != str]
    ppmSh = [k for k in ppmSh if type(k) != str]
    return shift, ppmSh

def calc_shift(max_avgs, ppms):
    shift = []
    ppmSh = []
    for i in range(len(max_avgs)):
        tTuple = (shift_tms(max_avgs[i], ppms[i]))
        shift.append(tTuple[0])
        ppmSh.append(tTuple[1])
    
    shift, ppmSh = rem_invalid_shifts(shift, ppmSh)
    
    return shift, ppmSh

def closer_to_zero(spec, ppm):
    cond = False
    max_index = np.where(spec == max(spec))[0][0]
    if(abs(ppm[max_index]) < (1 - abs(ppm[max_index]))):
        cond = True
    return cond

def focus_ppm_region(spec, ppm):
    #truncate TMS peak and water suppression peak 
    tempSpec = np.array(spec)
    index_of_TMS = np.where(tempSpec == max(tempSpec))[0][0]
    index_of_water = np.where(tempSpec == min(spec))[0][0]
    return spec[index_of_water: index_of_TMS], ppm[index_of_water:index_of_TMS]

def truncate(shift, ppmSh, bymax = True):
    truncated = shift[:]
    tPpm = ppmSh[:]
    
    if(bymax):
        maxlen = 0
        for i in range(len(truncated)):
            while(closer_to_zero(truncated[i], tPpm[i])):
                tTuple = focus_ppm_region(truncated[i], tPpm[i])
                truncated[i] = tTuple[0]
                tPpm[i] = tTuple[1]
            #print(f"Spectra #{i}:")
            #print(len(truncated[i]), len(tPpm[i]))
            if(len(truncated[i]) > maxlen):
                maxlen = len(truncated[i])
        
        for i in range(len(truncated)):
            if len(truncated[i]) != maxlen:
                #print(f"is not; {len(truncated[i])}")
                tpad = maxlen - len(truncated[i])
                truncated[i] = np.pad(truncated[i], (0, (maxlen - len(truncated[i]))))
                tPpm[i] = np.pad(tPpm[i], (0, (maxlen - len(tPpm[i]))))
            #print(len(truncated[i]), maxlen)
    else:
        minlen = math.inf
        for i in range(len(truncated)):
            while(closer_to_zero(truncated[i], tPpm[i])):
                tTuple = focus_ppm_region(truncated[i], tPpm[i])
                truncated[i] = tTuple[0]
                tPpm[i] = tTuple[1]
            #print(f"Spectra #{i}:")
            #print(len(truncated[i]), len(tPpm[i]))
            if(len(truncated[i]) < minlen):
                minlen = len(truncated[i])
        
        for i in range(len(truncated)):
            if len(truncated[i]) != minlen:
                #print(f"is not; {len(truncated[i])}")
                trem = minlen - len(truncated[i])
                truncated[i] = truncated[i][:trem]
                tPpm[i] = tPpm[i][:trem]
            #print(len(truncated[i]), maxlen)

    return truncated, tPpm
