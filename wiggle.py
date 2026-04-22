from myMolocule2 import *
import numpy as np
import random
import time
import copy

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

fileToRead = "leastNAD+.mol"
with open(fileToRead, 'r') as f:
  data = f.read()
  myMol = Molecule()
  myMol.molToVector(data, fileToRead)
  myMol.resetBondAngles('x')
  print(myMol)
  print(myMol.atoms[0])
  print(myMol.bonds[0])
  
  time1 = time.time()
  count = shiftRandom(myMol)
  print(myMol.bonds[0], "\nFound after " + str(count) + " itterations totalling " + str(time.time()-time1) + " seconds")
  myMol.saveMol()
  
    


