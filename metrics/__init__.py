from metrics import e_recall, c_recall, retrieval, openworld, lifelong
import numpy as np
import faiss, torch, copy
from sklearn.preprocessing import normalize
from tqdm import tqdm

def select(metricname):
    #### Metrics based on euclidean distances
    if 'e_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return e_recall.Metric(k)

    #### Metrics based on cosine similarity
    elif 'c_recall' in metricname:
        k = int(metricname.split('@')[-1])
        return c_recall.Metric(k)
    
    #### Retrieval metrics
    elif metricname == 'retrieval':
        return retrieval.RetrievalMetric()
    
    #### Open-world metrics
    elif metricname == 'openworld':
        return openworld.OpenWorldMetric()
    
    #### Lifelong metrics
    elif metricname == 'lifelong':
        return lifelong.LifelongMetric
    
    else:
        raise NotImplementedError("Metric {} not available!".format(metricname))
