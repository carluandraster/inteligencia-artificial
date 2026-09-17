"""Subtractive Clustering Algorithm
"""
__author__ = 'Daniel Albornoz'


from typing import cast

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist as distance_matrix

def subclust2(
    data: npt.NDArray[np.float64],
    Ra: float, Rb: float = 0,
    AcceptRatio: float = 0.3,
    RejectRatio: float = 0.1
)-> tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]:
    """Algoritmo de Clustering Substractivo
    
    Args:
        data (npt.NDArray[np.float64]): Datos de entrada. No puede ser None ni ser vacío.
        Ra (float): Radio de influencia para la densidad.
        Rb (float, optional): Radio de influencia para la distancia entre centros. Defaults to 0.
        AcceptRatio (float, optional): Ratio de aceptación para un nuevo centro. Defaults to 0.3.
        RejectRatio (float, optional): Ratio de rechazo para un nuevo centro. Defaults to 0.1.
    
    Returns:
        tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]: Etiquetas y centros del clustering."""
    
    if Rb==0:
        Rb = Ra*1.15

    scaler: MinMaxScaler = MinMaxScaler()
    scaler.fit(data)
    ndata: npt.NDArray[np.float64] = scaler.transform(data)

    potenciales: npt.NDArray[np.float64] = distance_matrix(ndata,ndata)
    alpha=(Ra/2)**2
    potenciales = np.sum(np.exp(-potenciales**2/alpha),axis=0)

    centers = []
    i=np.argmax(potenciales)
    c: npt.NDArray[np.float64] = ndata[i]
    p=potenciales[i]
    centers = [c]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            potenciales=potenciales-p*np.array([np.exp(-np.linalg.norm(v-c)**2/(Rb/2)**2) for v in ndata])
        restarP = True
        i=np.argmax(potenciales)
        c = ndata[i]
        p=potenciales[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,c))
        elif p<RejectRatio*pAnt:
            continuar=False
        else:
            dr = np.min([np.linalg.norm(v-c) for v in centers])
            if dr/Ra+p/pAnt>=1:
                centers = np.vstack((centers,c))
            else:
                potenciales[i]=0
                restarP = False
        if not any(v>0 for v in potenciales):
            continuar = False
    distancias: list[list[float]] = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(cast(npt.NDArray[np.float64], centers))
    return labels, centers