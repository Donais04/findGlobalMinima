from myMolocule import *
from scipy.optimize import minimize, basinhopping, differential_evolution, dual_annealing
import time
import numpy as np

class Algorithm():
    kwargs: dict
    mol: molocule
    alg: str
    result: float
    runTime: float
    name: str
    params: list[list]
    id: int
    stage: int
    skip: bool
    listType: str
    def __init__(self, mol: molocule, alg: str = "minimize", other: list[list] = [], 
                 name: str = "", id: int = -1, b: str = "", stage: int = 2, listType: str = "XYZ", skip1: bool = False, rand: bool = True):
        if stage == 2:
            print("Initializing algorithm " + alg + " with id " + str(id) + " with params " + str(other))
        self.listType = listType
        self.skip = skip1
        self.stage = stage
        self.params = other
        self.kwargs = {}
        self.mol = mol
        self.alg = alg
        self.name = name
        bounds = []
        if listType == "XYZ":
            self.listInFunc = self.mol.listToMineXYZ
            self.listOutFunc = self.mol.mineToListXYZ
            x0 = self.listOutFunc()
            bounds = [(-2.0, 2.0) for _ in range(len(x0))]
        elif listType == "PYM":
            self.listInFunc = self.mol.listToMine
            self.listOutFunc = self.mol.mineToList
            x0 = self.listOutFunc()
            for i in range(len(x0)):
                if (i+1) % 3 == 0:
                    bounds.append((0.5,2.0))
                else:
                    bounds.append((-1*2*np.pi,2*np.pi))
        elif listType == "FF":
            self.listInFunc = self.mol.zMatrixToMine
            self.listOutFunc = self.mol.mineToZMatrix
            x0 = self.listOutFunc()
            bounds = self.mol.zmatBounds()
        if not(b==""):
            self.batch = b
        else:
            self.batch = None
        if (id > 0):
            self.id = id
        if self.alg == "minimize":
            self.kwargs = {
                'fun': ObjectiveFunction(self.mol, self.listInFunc),
                'x0': x0,
                'method': 'L-BFGS-B',
                'tol': 1e-2,
                'options': {'maxiter': 1000, 'disp': False}
            }
            if self.kwargs['method'] in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
                self.kwargs['jac'] = '2-point'
        
        elif self.alg == "basinhopping":
            self.kwargs = {
                  'func': ObjectiveFunction(self.mol, self.listInFunc),
                  'x0': x0,
                  'T': 100.0,
                  'niter_success':5,
                  'disp':False
              }
        
        elif self.alg == "differential_evolution":
            self.kwargs = {
                'func': ObjectiveFunction(self.mol, self.listInFunc),
                'bounds': bounds,
                'x0': x0,
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
                'func': ObjectiveFunction(self.mol, self.listInFunc),
                'bounds': bounds,
                'x0': x0,
                'maxiter':1000,
                'minimizer_kwargs':None, 
                'initial_temp':5230,
                'restart_temp_ratio':2.e-5,
                'visit':2.62,
                'accept':-5.0,
                'maxfun':1e7,
                'no_local_search':False
                #'callback': callAn
            }
        
        elif self.alg == "random_restart":
            self.kwargs = {
                'func': ObjectiveFunction(self.mol, self.listInFunc),
                'bounds': bounds,
                'x0': x0,
                'iter': 20,
                'mini_iter': 100,
                'miniargs': {'method': 'L-BFGS-B',
                             'jac':'2-point','options': {}},
                'mol':self.mol,
                'listFunc':self.listType,
                'disp': False
            }
        
        elif self.alg == "brute_force":
            self.kwargs = {
                'func': ObjectiveFunction(self.mol, self.listInFunc),
                'bounds': bounds,
                'x0': x0,
                'iter': 1000
            }
            
        for i in other:
            self.kwargs[i[0]] = i[1]
        if stage == 1:
            if 'func' in self.kwargs.keys():
                self.kwargs['func'] = Stage1ObjectiveFunction(self.mol,self.listInFunc)
                self.kwargs['callback'] = callStage1
            else:
                self.kwargs['fun'] = Stage1ObjectiveFunction(self.mol,self.listInFunc)
                self.kwargs['callback'] = callStage1
    
    def run(self):
        if self.stage == 2:
            print("init " + str(self.mol.scoreFull()))
            print("running " + str(self.id))
            if not(self.skip):
                newAlg = Algorithm(self.mol,stage=1,listType="XYZ")
                newAlg.run()
                self.mol.listToMineXYZ(newAlg.mol.mineToListXYZ())
                self.kwargs['x0'] = self.listOutFunc()
                if 'func' in self.kwargs.keys():
                    print("starting from " + str(self.kwargs['func'](self.kwargs['x0'])))
                else:
                    print("starting from " + str(self.kwargs['fun'](self.kwargs['x0'])))
        #print(self.kwargs['x0'])
        if 'bounds' in self.kwargs.keys():
            for i in range(len(self.kwargs['x0'])):
                if self.kwargs['x0'][i] < self.kwargs['bounds'][i][0]:
                    self.kwargs['x0'][i] = self.kwargs['bounds'][i][0] + 0.01
                elif self.kwargs['x0'][i] > self.kwargs['bounds'][i][1]:
                    self.kwargs['x0'][i] = self.kwargs['bounds'][i][1] - 0.01
        
        self.listInFunc(self.kwargs['x0'])
        t1 = time.time()
        if self.alg == "minimize":
            list = minimize(**self.kwargs).x
        elif self.alg == "basinhopping":
            list = basinhopping(**self.kwargs).x
        elif self.alg == "differential_evolution":
            print(len(self.kwargs['x0']), len(self.kwargs['bounds']))
            list = differential_evolution(**self.kwargs).x
        elif self.alg == "dual_annealing":
            list = dual_annealing(**self.kwargs).x
        elif self.alg == "random_restart":
            list = random_restart(**self.kwargs)
        elif self.alg == "brute_force":
            list = brute_force(**self.kwargs)
        self.listInFunc(list)
        self.result = self.mol.scoreFull()
        self.runTime = time.time()-t1

    def save(self):
        builder = [["algorithm",self.alg],["runTime",self.runTime]]
        builder += self.params
        if self.batch:
            builder += ["batch",self.batch]
        self.mol.saveMol(other = builder, filename=self.name)
    
    def __str__(self) -> str:
        builder = "\n" + self.alg[0].upper().replace("_", " ") + self.alg[1:] + " algorithm " + str(self.id)
        if self.result:
            builder += " finished.\nResult: " + str(self.result) + "\nTook " + str(self.runTime) + " seconds."
        else:
            builder += " has not been ran yet"
        return builder + "\n"


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

def callStage1(x = 0, f = 0.0, context = 0):
    return f <= 0.0


class Stage1ObjectiveFunction:
    def __init__(self, mol: molocule, listfunc):
        self.listFunc = listfunc
        self.mol = mol
    
    def update(self,mol:molocule):
        self.mol = mol
    
    def __call__(self, coords: np.ndarray, scoreReqs: dict = {'minDistanceMult':1.2,'minAngle':np.pi/4}) -> float:
        self.listFunc(list(coords))
        return sum(self.mol.scoreValidity(**scoreReqs))

class ObjectiveFunction:
    def __init__(self, mol: molocule, listfunc):
        self.mol = mol
        self.listFunc = listfunc
    
    def update(self,mol:molocule):
        self.mol = mol
    
    def __call__(self, coords: np.ndarray) -> float:
        self.listFunc(list(coords))
        return self.mol.scoreFull()

def random_restart(mol: molocule, func, x0, miniargs, callback = None, bounds: list[tuple[float,float]] = [], 
                   iter: int = 200, mini_iter: int = 0, mini_tol: float = -1, disp: bool = False, listFunc: str = "XYZ") -> list[float]:
    if listFunc == "XYZ":
        funcIn = mol.listToMineXYZ
        funcOut = mol.mineToListXYZ
    elif listFunc == "PYM":
        funcIn = mol.listToMine
        funcOut = mol.mineToList
    elif listFunc == "FF":
        funcIn = mol.zMatrixToMine
        funcOut = mol.mineToZMatrix
        
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
        #print("starting", func(x0))
        #print(x0)
        #func.update(mol)
        miniargs['fun'] = func
        funcIn(minimize(**miniargs).x)
        score = func(funcOut())
        updated = False
        if score < bestScore:
            bestScore = score
            bestList = funcOut()
            updated = True
        if callback:
            if callback(x0,score,updated):
                return bestList
        for i in range(len(x0)):
            x0[i] = (random.random() * (bounds[i][1]-bounds[i][0])) + bounds[i][0]
        funcIn(x0)
        newAlg = Algorithm(mol,stage=1)
        newAlg.run()
        if disp:
            dispRandRest(x0, score, updated)
            print("starting from " + str(mol.scoreFull()))
    return bestList


def brute_force(func, x0, bounds: list[tuple[float,float]], callback = None, iter: int = 200, disp:bool = False):
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
            dispRandRest(x0, score, updated)
        if callback:
            if callback(x0,score,updated):
                return bestList
    return bestList












