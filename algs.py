from myMolocule import *
from scipy.optimize import minimize, basinhopping, differential_evolution, dual_annealing
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
    id: int
    def __init__(self, mol: molocule, alg: str = "minimize", other: list[list] = [], name: str = "", id: int = -1):
        print("Initializing algorithm " + alg + " with id " + str(id) + " with params " + str(other))
        self.params = other
        self.kwargs = {}
        self.mol = mol
        self.alg = alg
        self.name = name
        if (id > 0):
            self.id = id
        
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
                  'T': 100.0,
                  'niter_success':5,
                  'disp':False
              }
        
        elif self.alg == "differential_evolution":
            self.kwargs = {
                'func': ObjectiveFunction(self.mol),
                'bounds': [(-2.0,2.0) for i in range(len(self.mol.mineToListXYZ()))],
                'x0': self.mol.mineToListXYZ(),
                'args': (),
                'strategy': 'best1bin',
                'maxiter': 1000,
                'popsize': 10,
                'tol': 1e-3,
                'mutation': 0.5,
                'recombination': 0.7,
                'seed': None,
                'disp': True,
                'workers': 6,
                'updating': 'deferred'
            }
        
        elif self.alg == "dual_annealing":
            self.kwargs = {
                'func': ObjectiveFunction(self.mol),
                'bounds': [(-2.0,2.0) for i in range(len(self.mol.mineToListXYZ()))],
                'x0': self.mol.mineToListXYZ(),
                'args': ()#,
                #'callback': callAn
            }
        for i in other:
            self.kwargs[i[0]] = i[1]
    
    def run(self):
        print("running " + str(self.id))
        t1 = time.time() 
        if self.alg == "minimize":
            self.mol.listToMineXYZ(minimize(**self.kwargs).x)
        elif self.alg == "basinhopping":
            self.mol.listToMineXYZ(basinhopping(**self.kwargs).x)
        elif self.alg == "differential_evolution":
            self.mol.listToMineXYZ(differential_evolution(**self.kwargs).x)
        elif self.alg == "dual_annealing":
            self.mol.listToMineXYZ(dual_annealing(**self.kwargs).x)
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


def callAn(x, f, context):
    print("score " + str(f) + " with context ", end="")
    if context == 0:
        print("minimum detected in the annealing process")
    elif context == 1:
        print("detection occurred in the local search process")
    elif context == 2:
        print("detection done in the dual annealing process")


def objective(coords, mola: molocule):
    mola.listToMineXYZ(coords)
    return mola.scoreFull()

class ObjectiveFunction:
    def __init__(self, mol: molocule):
        self.mol = mol
    
    def __call__(self, coords: np.ndarray) -> float:
        self.mol.listToMineXYZ(list(coords))
        return self.mol.scoreFull()

