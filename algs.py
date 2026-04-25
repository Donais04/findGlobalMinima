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
    name: str
    params: list[list]
    def __init__(self, mol: molocule, alg: str = "minimize", other: list[list] = [], name: str = ""):
        print("Initializing algorithm " + alg)
        self.params = other
        self.kwargs = {}
        self.mol = mol
        self.alg = alg
        self.name = name
        if self.alg == "minimize":
            self.kwargs = {
                'fun': lambda coords: objective(coords, self.mol),
                'x0': self.mol.mineToListXYZ(),
                'method': 'L-BFGS-B',
                'tol': 1e-2,
                'options': {'maxiter': 1000, 'disp': False}
            }
            if self.kwargs['method'] in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
                self.kwargs['jac'] = '2-point'
        
        elif self.alg == "basinhopping":
            self.kwargs = {
                  'func': lambda coords: objective(coords, self.mol),
                  'x0': self.mol.mineToListXYZ(),
                  'T': 20.0,
                  'niter_success':5,
                  'disp':False
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
                'disp': False,
                'workers': 4,
                'updating': 'deferred'
            }
        for i in other:
            self.kwargs[i[0]] = i[1]
    
    def run(self):
        t1 = time.time() 
        print("running " + self.name)
        if self.alg == "minimize":
            self.mol.listToMineXYZ(minimize(**self.kwargs).x)
        elif self.alg == "basinhopping":
            self.mol.listToMineXYZ(basinhopping(**self.kwargs).x)
        elif self.alg == "differential_evolution":
            self.mol.listToMineXYZ(differential_evolution(**self.kwargs).x)
        self.result = self.mol.scoreFull()
        self.runTime = time.time()-t1

    def save(self):
        builder = [["algorithm",self.alg],["runTime",self.runTime]]
        builder += self.params
        self.mol.saveMol(other = builder, filename=self.name)
    
    def __str__(self) -> str:
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

