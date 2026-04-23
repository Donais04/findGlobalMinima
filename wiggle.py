from myMolocule import *
from algs import *
import numpy as np
import random
import time
import copy
from scipy.optimize import minimize, basinhopping


_FILE_TO_READ = "lessNAD+.mol"

def fullRandom(myMol):
  count = 0
  while not(myMol.scoreValidity(shifting=True, minDistanceMult=0.6)):
    count += 1
    print(count)
    for i in myMol.bonds:
      i.setBondPitch((random.random() - 0.5) * 2 * np.pi * 2)
      i.setBondYaw((random.random() - 0.5) * 2 * np.pi * 2)
      
  return count
      
      
      
def shiftRandom(myMol):
  count = 0
  best = myMol
  bestA = 100000.0
  while True:
    count += 1
    score = myMol.scoreValidity(minDistanceMult=1)
    print(count)
    if score < bestA:
      bestA = score
      best = copy.deepcopy(myMol)
    if score < 0.01:
      break
    for i in myMol.bonds:
      i.setBondPitch((random.random() - 0.5) * 2 * np.pi / 2)
      i.setBondYaw((random.random() - 0.5) * 2 * np.pi / 2)
    if count == 200:
      return best

def norm():
  print("start")
  molString = open(_FILE_TO_READ).read()
  mol = molocule()
  mol.molToVector(molString, _FILE_TO_READ)
  mol.resetBondAngles()
  mol.rando()

  def objective(coords, mola):
      mola.listToVectorXYZ(coords)
      return mola.scoreValidity(minDistanceMult = 1.3)


  kwargs = {
          'fun': lambda coords: objective(coords, mol),
          'x0': mol.vectorToListXYZ(),
          'method': 'BFGS',
          'tol': 1e-6,
          'options': {'maxiter': 1000, 'disp': True}
      }

  if kwargs['method'] in ['CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']:
          kwargs['jac'] = '2-point'

  result = minimize(**kwargs)

  #print(result.x)

  mol.listToVectorXYZ(result.x)

  print(mol.scoreValidity())

  mol.saveMol()

def notnorm():
  print("start")
  molString = open(_FILE_TO_READ).read()
  mol = molocule()
  mol.molToVector(molString, _FILE_TO_READ)
  mol.resetBondAngles()
  mol.rando()

  def objective(coords, mola: molocule):
      mola.listToVectorXYZ(coords)
      return mola.scoreValidity(minDistanceMult = 1.5,power = 5)


  kwargs = {
          'func': lambda coords: objective(coords, mol),
          'x0': mol.vectorToListXYZ(),
          'T': 1.0,
          'niter_success':5,
          'disp':True
      }
  
  result = basinhopping(**kwargs)

  mol.listToVectorXYZ(result.x)

  print(mol.scoreValidity())

  mol.saveMol()
notnorm()
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