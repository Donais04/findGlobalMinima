from myMolocule2 import *
import random
from scipy.optimize import minimize

class Algorithm():
  bestScore: float
  bestMol: Molecule
  ID: int
  mol: Molecule
  num: int
  
  def __init__(self, m: Molecule, n: int = 0, bs: int = 100000, id: int = -1):
    self.mol = m
    self.bestMol = m
    self.bestScore = bs
    self.num = n
    if id < 0:
      self.ID = int(random.random() * 922337203685477580)
    else:
      self.ID = id
  
  def itterate(self) -> Molecule:
    self.num += 1
    return self.mol


