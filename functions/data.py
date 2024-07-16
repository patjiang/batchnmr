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
            if("." not in file_arr[k]):
                temp_dict[file_arr[k]] = nmrpy.from_path(fid_path=f"./{file_dir}/{exp_names[j]}/{key}/{file_arr[k]}")
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
