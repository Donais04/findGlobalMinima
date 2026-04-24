from myMolocule import *
from algs import *
import numpy as np
import random, os, sys
import copy
from scipy.optimize import minimize, basinhopping
import asyncio


_FILE_TO_READ = "leastNAD+.mol"

async def runXTimesLimitedConcurrency(x: int, mol, maxConcurrent: int = 10): #AI written
    """
    Run function x times with limited concurrency.

    Parameters
    ----------
    x : int
        Number of times to run.
    maxConcurrent : int, optional
        Maximum concurrent tasks. Defaults to 10.

    Returns
    -------
    list
        List of results.
    """
    semaphore = asyncio.Semaphore(maxConcurrent)

    async def limitedRun():
        async with semaphore:
            return await norm(mol)

    tasks = [limitedRun() for _ in range(x)]
    results = await asyncio.gather(*tasks)
    return results


async def norm(mol):
  print("start")

  def objective(coords, mola):
      mola.listToVectorXYZ(coords)
      return mola.scoreFull()


  kwargs = {
          'fun': lambda coords: objective(coords, mol),
          'x0': mol.vectorToListXYZ(),
          'method': 'BFGS',
          'tol': 1e-2,
          'options': {'maxiter': 1000, 'disp': True}
      }

  if kwargs['method'] in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
          kwargs['jac'] = '2-point'

  result = minimize(**kwargs)

  #print(result.x)

  mol.listToVectorXYZ(result.x)
  
  mol.saveMol()
  print(mol.scoreFull())

async def notnorm(mol):
  print("start")

  def objective(coords, mola: molocule):
      mola.listToVectorXYZ(coords)
      return mola.scoreFull()


  kwargs = {
          'func': lambda coords: objective(coords, mol),
          'x0': mol.vectorToListXYZ(),
          'T': 20.0,
          'niter_success':5,
          'disp':True
      }
  
  result = basinhopping(**kwargs)

  mol.listToVectorXYZ(result.x)

  mol.saveMol()
  print(mol.scoreFull())




async def func(t, mol):
  await runXTimesLimitedConcurrency(t, mol, maxConcurrent=4)


myMol = molocule()
myMol.molToVector(open(_FILE_TO_READ).read())

if __name__ == "__main__":
    t = int(sys.argv[1])
    print(t)
    for i in range(t):
      myMol = molocule()
      myMol.molToVector(open(_FILE_TO_READ).read())
      myMol.rando()
      asyncio.run(norm(myMol))
    
    
    
    
    
# Minimize
#minimized = mol.minimizeOpenFF()
#print(f"Minimized score: {minimized.scoreValidity()}")



#mola = norm()
#print(mola.scoreValidity())
#mola.saveMol(fileType="pdb")

## Create initial molecule from mol file
#molString = open(_FILE_TO_READ).read()
#mol = Molecule()
#mol.molToVector(molString, _FILE_TO_READ)
#
## Initialize basin hopping
#optimizer = BasinHopping(
#    mol,
#    temperature=1000.0,
#    stepSize=0.5,
#    maxIterations=10000
#)
#
## Run with progress callback
#def progressCallback(iteration, score, temperature):
#    if iteration % 100 == 0:
#        print(f"Iter {iteration}: score={score:.4f}, T={temperature:.4f}")
#    return True  # Continue optimizing
#
#bestMolecule = optimizer.run(callback=progressCallback)
#
#print(bestMolecule.scoreValidity())
#
## Print final statistics
#print(optimizer.getStatistics())
#print(f"Best score: {optimizer.bestScore}")