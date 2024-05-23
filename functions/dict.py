'''
Similar to the other data importing file, this one presumes a familiarity to dictionary architecture
JSON file saving format
'''
def fidDictViz(dataDict):
    if type(dataDict) is dict:
        for i in dataDict.keys():
            print(f"{i}:")
            print(f"\t{fidDictViz(dataDict[i])}")
    #type(a) is nmrpy.data_objects.FidArray
        return ""
    elif type(dataDict) is np.ndarray:
        print(f"\t{dataDict}")
        return ""
    else:
        if(len(dataDict.get_fids()) > 10):
            print(f"\t{[f.id for f in dataDict.get_fids()[1:5]]}, {len(dataDict.get_fids()) - 4} keys omitted")
        else:
            print(f"\t{[f.id for f in dataDict.get_fids()]}")
        return ""

def preproc_max_inDict(dataDict):
    for i in dataDict.keys():
        tAvg = {}
        tPpm = {}
        for j in dataDict[i].keys():
            preprocess(dataDict[i][j])
            tTuple = avg_spectra(dataDict[i][j], len(dataDict[i][j].data))
            tAvg[j] = tTuple[0]
            tPpm[j] = tTuple[1]
        dataDict[i]["max_avgs"] = tAvg
        dataDict[i]["max_ppms"] = tPpm

def plotmaxDict(dataDict, samples, cmap = 'viridis'):
    map = matplotlib.colormaps[cmap].resampled(samples)
    counter = 0
    for i in dataDict.keys():
        for j in dataDict[i]["max_avgs"].keys():
            plt.plot(dataDict[i]["max_ppms"][j], dataDict[i]["max_avgs"][j],
                     c = map.colors[counter], 
                     label = f"{i}_{j}")
            counter += 1
    plt.legend()
    plt.xlim(left = -1, right = 6)


def shift_tms(fid_arr, ppm, verbose):
    pos = np.where(fid_arr == max(fid_arr))
    zero = np.where((ppm == min(abs(ppm))) | (ppm == -1 * min(abs(ppm))))
    shift = True
    if(pos != zero):
        if verbose:
            print(f"\tPosition Not Equal: \tfidpos: {pos[0][0]}\tppmpos: {zero[0][0]}, \tppm value: {ppm[zero[0][0]]}\n")
    else:
        if verbose:
            print(f"\tPosition Equal: \tfidpos: {pos[0][0]}\tppmpos: {zero[0][0]}, \tppm value: {ppm[zero[0][0]]}\n")
        shift = False
        newFID = fid_arr

    if shift:
        if verbose:
            print("\tShifting fid such that the highest frequency peak is also closest to 0\n")
        diff = zero[0][0] - pos[0][0]
        if verbose:
            print(f"\tShift amount = {diff}\n")
        if (diff > (len(fid_arr)/10)):
            if verbose:
                print("\tToo much shift. Alignment Failed\n")
            return "emp","ty"
        else:
            newFID = np.roll(fid_arr, diff)
            shift_tms(newFID, ppm, verbose = verbose)

    #set min ppm to 0
    
    ppmval = ppm[zero]

    new_ppm = ppm - ppmval

    pos = np.where(newFID == max(newFID))
    zero = np.where((new_ppm == min(abs(new_ppm))) | (new_ppm == -1 * min(abs(new_ppm))))
        
    if verbose:
        print(f"final values: \tfidpos: {pos[0][0]}\tppmpos: {zero[0][0]}, \tppm value: {new_ppm[zero[0][0]]}\n")
    
    return newFID, new_ppm

def rem_invalid_shift_dict(shift, ppmSh):
    nshift = dict(shift)
    nppmSh = dict(ppmSh)
    for i in shift.keys():
        if type(nshift[i]) == str:
            del nshift[i]
            del nppmSh[i]
    return nshift, nppmSh

def calc_shift_dict(dataDict, verbose = True):
    for i in dataDict.keys():
        tshift = {}
        tppms = {}
        for j in dataDict[i]["max_avgs"].keys():
            tTuple = (shift_tms(dataDict[i]["max_avgs"][j], dataDict[i]["max_ppms"][j], verbose = verbose))
            tshift[j] = tTuple[0]
            tppms[j] = tTuple[1]
        tshift, tppms = rem_invalid_shift_dict(tshift, tppms)
        dataDict[i]["shift"] = tshift
        dataDict[i]["ppmSh"] = tppms

def plotshiftDict(dataDict, samples, cmap = 'viridis'):
    map = matplotlib.colormaps[cmap].resampled(samples)
    counter = 0
    for i in dataDict.keys():
        for j in dataDict[i]["shift"].keys():
            plt.plot(dataDict[i]["ppmSh"][j], dataDict[i]["shift"][j],
                     c = map.colors[counter], 
                     label = f"{i}_{j}")
            counter += 1
    plt.legend()
    plt.xlim(left = -1, right = 6)


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

def repad(spec, ppm, n, max = True):
    tspec = spec
    tppm = ppm
    if max:
        if(len(tspec) < n):
            tspec = np.pad(tspec, (0, n - len(tspec)))
            tppm = np.pad(tppm, (0, n - len(tppm)))
    else:
        if(len(spec) > n):
            tspec = tspec[:(n - len(tspec))]
            tppm = tppm[:(n - len(tppm))]
    
    return tspec, tppm
        

def truncate_dict(dataDict, bymax = True):
    if(bymax):
        nlen = 0
    else:
        nlen = math.inf
    for i in dataDict.keys():
        trun = {}
        tppm = {}
        for j in dataDict[i]["shift"]:
            #print(f"Truncating shifted spectra in {j}")
            trun[j] = dataDict[i]["shift"][j]
            tppm[j] = dataDict[i]["ppmSh"][j]
            while(closer_to_zero(trun[j], tppm[j])):
                tTuple = focus_ppm_region(trun[j], tppm[j])
                trun[j] = tTuple[0]
                tppm[j] = tTuple[1]
            if(len(trun[j]) > nlen) and bymax:
                nlen = len(trun[j])
            elif(len(trun[j]) < nlen) and not bymax:
                nlen = len(trun[j])
        dataDict[i]["truncated"] = trun
        dataDict[i]["tPpm"] = tppm
    #print(nlen)

    for i in dataDict.keys():
        for k in dataDict[i]["truncated"].keys():
            dataDict[i]["truncated"][k], dataDict[i]["tPpm"][k] = repad(dataDict[i]["truncated"][k], dataDict[i]["tPpm"][k], n = nlen, max = bymax)
            #print(len(dataDict[i]["truncated"][k]), len(dataDict[i]["tPpm"][k]))

def plotTruncdict(dataDict, samples, cmap = 'viridis'):
    map = matplotlib.colormaps[cmap].resampled(samples)
    counter = 0
    for i in dataDict.keys():
        for j in dataDict[i]["truncated"].keys():
            #print(len(dataDict[i]["tPpm"][j]), len(dataDict[i]["tPpm"][j]))
            plt.plot(dataDict[i]["tPpm"][j], dataDict[i]["truncated"][j],
                     c = map.colors[counter], 
                     label = f"{i}_{j}")
            counter += 1
    plt.legend()
