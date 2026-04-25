from myMolocule import *
from scipy.optimize import minimize, basinhopping, differential_evolution, Bounds
import asyncio, time
import numpy as np
from collections.abc import Callable

class Algorithm():
    kwargs: dict
    mol: molocule
    alg: str
    result: float
    runTime: float
    def __init__(self, mol: molocule, alg: str = "minimize"):
        print("Initializing algorithm " + alg)
        self.kwargs = {}
        self.mol = mol
        self.alg = alg
        if self.alg == "minimize":
            self.kwargs = {
                'fun': lambda coords: objective(coords, self.mol),
                'x0': self.mol.mineToListXYZ(),
                'method': 'TNC',
                'tol': 1e-2,
                'options': {'maxiter': 1000, 'disp': True}
            }
            if self.kwargs['method'] in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
                self.kwargs['jac'] = '2-point'
        
        elif self.alg == "basinhopping":
            self.kwargs = {
                  'func': lambda coords: objective(coords, self.mol),
                  'x0': self.mol.mineToListXYZ(),
                  'T': 20.0,
                  'niter_success':5,
                  'disp':True
              }
        
        elif self.alg == "differential_evolution":
            self.kwargs = {
                'func': ObjectiveFunction(self.mol),
                'bounds': [(-2.0,2.0) for i in range(len(self.mol.mineToListXYZ()))],
                'args': (),
                'strategy': 'best1bin',
                'maxiter': 1000,
                'popsize': 15,
                'tol': 1e-3,
                'mutation': 0.5,
                'recombination': 0.7,
                'seed': None,
                'disp': True,
                'workers': 4,
                'updating': 'deferred'
            }
    
    def run(self):
        t1 = time.time() 
        if self.alg == "minimize":
            self.mol.listToMineXYZ(minimize(**self.kwargs).x)
        elif self.alg == "basinhopping":
            self.mol.listToMineXYZ(basinhopping(**self.kwargs).x)
        elif self.alg == "differential_evolution":
            self.mol.listToMineXYZ(differential_evolution(**self.kwargs).x)
        self.result = self.mol.scoreFull()
        self.runTime = time.time()-t1

    def save(self):
        self.mol.saveMol(other = [["algorithm",self.alg],["runTime",self.runTime]])
    
    def str(self) -> str:
        builder = self.alg[0].upper() + self.alg[1:] + " algorithm.\n"
        if self.result:
            builder += "Result: " + str(self.result)
        else:
            builder += "Has not been ran yet"
        return builder



def objective(coords, mola: molocule):
    mola.listToMineXYZ(coords)
    return mola.scoreFull()

class ObjectiveFunction:
    def __init__(self, mol: molocule):
        self.mol = mol
    
    def __call__(self, coords: np.ndarray) -> float:
        self.mol.listToMineXYZ(list(coords))
        return self.mol.scoreFull()

