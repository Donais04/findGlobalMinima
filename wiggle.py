from myMolocule import *
import numpy as np
import random
import time

def fullRandom(myMol):
  count = 0
  while not(myMol.getValidity(shifting=True, minDistanceMult=0.6)):
    count += 1
    print(count)
    for i in myMol.bonds:
      i.setBondPitch((random.random() - 0.5) * 2 * np.pi * 2)
      i.setBondYaw((random.random() - 0.5) * 2 * np.pi * 2)
      
  return count
      
def shiftRandom(myMol):
  count = 0
  
  while True:
    count += 1
    score = myMol.scoreValidity(minDistanceMult=0.8)
    print(score)
    if score < 0.1:
      break
    for i in myMol.bonds:
      i.setBondPitch((random.random() - 0.5) * 2 * np.pi / 2)
      i.setBondYaw((random.random() - 0.5) * 2 * np.pi / 2)

with open("compChem/leastNAD+.mol", 'r') as f:
  data = f.read()
  myMol = molocule()
  myMol.molToVector(data)
  myMol.resetBondAngles('x')
  print(myMol)
  print(myMol.atoms[0])
  print(myMol.bonds[0])
  
  time1 = time.time()
  count = shiftRandom(myMol)
  print(myMol.bonds[0], "\nFound after " + str(count) + " itterations totalling " + str(time.time()-time1) + " seconds")
  with open("compChem/out.mol", 'w') as g:
    g.write(myMol.vectorToMol())
    


