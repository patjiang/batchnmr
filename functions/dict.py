'''
Similar to the other data importing file, this one presumes a familiarity to dictionary architecture
JSON file saving format
'''
import numpy as np
import nmrpy

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

def plotmaxDict(dataDict, samples, cmap = 'viridis'):
    map = matplotlib.colormaps[cmap].resampled(samples)
    for i in dataDict.keys():
        for j in dataDict[i]["max_avgs"].keys():
            plt.plot(dataDict[i]["max_ppms"][j], dataDict[i]["max_avgs"][j],
                     c = map.colors[samples - 1], 
                     label = f"{i}_{j}")
            samples -= 1
    plt.legend()
    plt.xlim(left = -1, right = 6)
