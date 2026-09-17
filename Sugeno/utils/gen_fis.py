# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:56:16 2020

@author: Daniel Albornoz

Implementación similar a genfis2 de Matlab.
Sugeno type FIS. Generado a partir de clustering substractivo.

"""
__author__ = 'Daniel Albornoz'

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time
from collections.abc import Callable

from utils.clustering_substractivo import subclust2

def gaussmf(data: npt.NDArray[np.float64],
            mean: npt.NDArray[np.float64],
            sigma: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Función de membresía gaussiana.
    
    Args:
        data (npt.NDArray[np.float64]): Datos de entrada.
        mean (npt.NDArray[np.float64]): Media de la distribución gaussiana.
        sigma (npt.NDArray[np.float64]): Desviación estándar de la distribución gaussiana.

    Returns:
        npt.NDArray[np.float64]: Valores de membresía para cada dato de entrada.
    """
    exp_val = -((data - mean)**2.) / (2 * sigma**2.)
    # fix de tipos:
    exp_val = exp_val.astype(np.float32)
    return np.exp( exp_val )

class fisRule:
    """Clase que representa una regla de inferencia difusa.
    
    Attributes:
        centroid (npt.NDArray[np.float64]): Centroide de la regla.
        sigma (npt.NDArray[np.float64]): Desviación estándar de la regla.
    """
    def __init__(self,
                 centroid: npt.NDArray[np.float64],
                 sigma: npt.NDArray[np.float64]
    ):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    """Clase que representa una entrada de un sistema de inferencia difusa.
    
    Attributes:
        minValue (npt.NDArray[np.float64]): Valor mínimo de la entrada.
        maxValue (npt.NDArray[np.float64]): Valor máximo de la entrada.
        centroids (npt.NDArray[np.float64]): Centroides de las funciones de membresía.
    
    Methods:
        view(): Visualiza las funciones de membresía de la entrada.
    """
    def __init__(self,
                 min: npt.NDArray[np.float64],
                 max: npt.NDArray[np.float64],
                 centroids: npt.NDArray[np.float64]
    ):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids

    def view(self) -> None:
        """Visualiza las funciones de membresía de la entrada."""
        x = np.linspace(self.minValue,self.maxValue,20)
        plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
            plt.plot(x,y)

class fis:
    """Clase que representa un sistema de inferencia difusa tipo Sugeno.
    
    Attributes:
        rules (list): Lista de reglas del sistema.
        memberfunc (list): Lista de funciones de membresía del sistema.
        inputs (list): Lista de entradas del sistema.
        solutions (list): Lista de soluciones del sistema.
    Métodos:
        genfis(data, radii): Genera el sistema de inferencia difusa a partir de los datos y un parámetro de radio.
        entrenar(data): Entrena el sistema de inferencia difusa utilizando mínimos cuadrados.
        evalfis(data): Evalúa el sistema de inferencia difusa.
        viewInputs(): Visualiza las funciones de membresía de todas las entradas del sistema.
    """
        
    rules: npt.NDArray[np.float64]
    memberfunc: list[np.float64]
    inputs: list[fisInput]
    solutions: npt.NDArray[np.float32]
    
    def __init__(self):
        self.rules=np.array([])
        self.memberfunc = []
        self.inputs = []

    def genfis(self,
               data: npt.NDArray[np.float64],
               clustering: Callable[..., tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]] = subclust2
    ) -> None:
        """Genera el sistema de inferencia difusa a partir de los datos y un parámetro de radio.
        
        Args:
            data (npt.NDArray[np.float64]): Datos de entrada para generar el sistema.
            clustering (Callable[..., tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]]): Función de clustering. Default es subclust2.
        """
        start_time = time.time()
        _, cluster_center = clustering()

        print("--- %s seconds ---" % (time.time() - start_time))

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self,
                 data: npt.NDArray[np.float64]
    ) -> int:
        """Entrena el sistema de inferencia difusa utilizando mínimos cuadrados.
        
        Args:
            data (npt.NDArray[np.float64]): Datos de entrada para entrenar el sistema.
        Returns:
            int: El número de reglas entrenadas.
        """
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        print("nivel acti")
        print(nivel_acti)
        sumMu = np.sum(nivel_acti, axis=1, keepdims=True)
        print("sumMu")
        print(sumMu)
        p = np.c_[P, np.ones(len(P))]
        n_vars = p.shape[1]

        orden = np.tile(np.arange(0,n_vars, dtype=np.intp), len(self.rules))
        acti = np.tile(nivel_acti, (1, n_vars))
        inp = p[:, orden]


        A = acti*inp/sumMu

        # fix de tipos:
        a: npt.NDArray[np.float32] = A.astype(np.float32)
        b: npt.NDArray[np.float32] = T.astype(np.float32)

        solutions: npt.NDArray[np.float32] = np.asarray(
            np.linalg.lstsq(a, b, rcond=None)[0],
            dtype=np.float32
        )

        self.solutions = solutions
        print(solutions)

        return len(self.rules)

    def evalfis(self,
                data: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Evalúa el sistema de inferencia difusa.

        Args:
            data (npt.NDArray[np.float64]): Datos de entrada para evaluar el sistema.

        Returns:
            npt.NDArray[np.float64]: Resultados de la evaluación.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.sum(nivel_acti, axis=1, keepdims=True)

        P: npt.NDArray[np.float64] = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars, dtype=np.intp), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        """Visualiza las funciones de membresía de todas las entradas del sistema."""
        for input in self.inputs:
            input.view()