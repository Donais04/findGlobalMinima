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
    def __init__(self, mol: molocule, alg: str = "minimize", other: list[list] = [], name: str = "", id: int = -1, b: str = "", stage: int = 2):
        print("Initializing algorithm " + alg + " with id " + str(id) + " with params " + str(other))
        self.params = other
        self.kwargs = {}
        self.mol = mol
        self.alg = alg
        self.name = name
        if not(b==""):
            self.batch = b
        else:
            self.batch = None
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
                'strategy': 'best1bin',
                'maxiter': 1000,
                'popsize': 10,
                'tol': 1e-3,
                'mutation': 0.5,
                'recombination': 0.7,
                'disp': False,
                'workers': 1,
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
        
        elif self.alg == "random_restart":
            self.kwargs = {
                'func': Stage1ObjectiveFunction(self.mol),
                'bounds': [(-2.0,2.0) for i in range(len(self.mol.mineToListXYZ()))],
                'x0': self.mol.mineToListXYZ(),
                'iter': 20,
                'mini_iter': 100,
                'miniargs': {'method': 'L-BFGS-B',
                             'jac':'2-point','options': {}},
                'mol':self.mol,
                'disp': False
            }
        
        elif self.alg == "brute_force":
            self.kwargs = {
                'func': lambda coords: objective(coords, self.mol),
                'bounds': [(-2.0,2.0) for _ in range(len(self.mol.mineToListXYZ()))],
                'x0': self.mol.mineToListXYZ(),
                'iter': 1000
            }
            
        for i in other:
            self.kwargs[i[0]] = i[1]
        if stage == 1:
            if 'func' in self.kwargs.keys():
                self.kwargs['func'] = Stage1ObjectiveFunction(self.mol)
                self.kwargs['callback'] = callStage1
            else:
                self.kwargs['fun'] = Stage1ObjectiveFunction(self.mol)
                self.kwargs['callback'] = callStage1
    
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
        elif self.alg == "random_restart":
            self.mol.listToMineXYZ(random_restart(**self.kwargs))
        elif self.alg == "brute_force":
            self.mol.listToMineXYZ(brute_force(**self.kwargs))
        self.result = self.mol.scoreFull()
        self.runTime = time.time()-t1

    def save(self):
        builder = [["algorithm",self.alg],["runTime",self.runTime]]
        builder += self.params
        if self.batch:
            builder += ["batch",self.batch]
        self.mol.saveMol(other = builder, filename=self.name)
    
    def __str__(self) -> str:
        builder = self.alg[0].upper().replace("_", " ") + self.alg[1:] + " algorithm " + str(self.id)
        if self.result:
            builder += " finished.\nResult: " + str(self.result) + "\nTook " + str(self.runTime) + " seconds."
        else:
            builder += " has not been ran yet"
        return builder


def callAn(x, f, context):
    print("score " + str(f) + " with context ", end="")
    if context == 0:
        print("minimum detected in the annealing process")
    elif context == 1:
        print("detection occurred in the local search process")
    elif context == 2:
        print("detection done in the dual annealing process")

def dispRandRest(x, f, context):
    print("score " + str(f) + " with context ", end="")
    if context:
        print("minimum detected")
    else:
        print("no minimum obtained")

def callStage1(x, f, context):
    return f <= 0.0

class Stage1ObjectiveFunction:
    def __init__(self, mol: molocule):
        self.mol = mol
    
    def update(self,mol:molocule):
        self.mol = mol
    
    def __call__(self, coords: np.ndarray, scoreReqs: dict = {}) -> float:
        self.mol.listToMineXYZ(list(coords))
        return sum(self.mol.scoreValidity(**scoreReqs))
    
def stage1Objective(coords, mol: molocule, scoreReqs: dict = {}):
    mol.listToMineXYZ(list(coords))
    return sum(mol.scoreValidity(**scoreReqs))

def objective(coords, mol: molocule):
    mol.listToMineXYZ(coords)
    return mol.scoreFull()

class ObjectiveFunction:
    def __init__(self, mol: molocule):
        self.mol = mol
    
    def update(self,mol:molocule):
        self.mol = mol
    
    def __call__(self, coords: np.ndarray) -> float:
        self.mol.listToMineXYZ(list(coords))
        return self.mol.scoreFull()

def random_restart(mol: molocule, func, x0, miniargs, callback = None, bounds: list[tuple[float,float]] = [], iter: int = 200, mini_iter: int = 0, mini_tol: float = -1, disp: bool = False) -> list[float]:
    if bounds == []:
        for _ in x0:
            bounds.append((-10000.0,10000.0))
    miniargs['options']['maxiter'] = mini_iter
    bestScore: float = 9999999999.0
    bestList: list[float] = x0
    if mini_tol > 0:
        miniargs['tol'] = mini_tol
    for _ in range(iter):
        miniargs['x0'] = x0
        func.update(mol)
        miniargs['fun'] = func
        mol.listToMineXYZ(minimize(**miniargs).x)
        score = func(mol.mineToListXYZ())
        updated = False
        if score < bestScore:
            bestScore = score
            bestList = mol.mineToListXYZ()
            updated = True
        for i in range(len(x0)):
            x0[i] = (random.random() * (bounds[i][1]-bounds[i][0])) + bounds[i][0]
        if disp:
            dispRandRest(x0, score, updated)
        if callback:
            if callback(x0,score,updated):
                return bestList
    return bestList


def brute_force(func, x0, bounds: list[tuple[float,float]], iter: int = 200, disp:bool = False):
    if bounds == []:
        for _ in x0:
            bounds.append((-10000.0,10000.0))
    bestScore: float = 9999999999.0
    bestList: list[float] = x0
    for _ in range(iter):
        score = func(x0)
        updated = False
        if score < bestScore:
            bestScore = score
            bestList = x0
            updated = True
        for i in range(len(x0)):
            x0[i] = (random.random() * (bounds[i][1]-bounds[i][0])) + bounds[i][0]
        if disp:
            callRandRest(x0, score, updated)
    return bestList