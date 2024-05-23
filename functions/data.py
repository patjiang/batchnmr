```
This file is mainly about importing data and preprocessing it through nmrpy; for the base functions, it is recommended that you check out their repo
https://github.com/NMRPy/nmrpy
```

import nmrpy
import time
import numpy as np

```
Please read the maine page or the readme for an explanation on the data format requirements. Alternatively, check out the ipynb for examples
```
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
