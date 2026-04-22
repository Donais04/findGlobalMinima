import numpy as np
import random
import math

class bond():
  bondType: int
  vector: list[float]
  yaw: float
  pitch: float
  def __init__(self,f,t,n):
    self.atomFrom = f
    self.atomTo = t
    self.bondType = n
    self.vector = [t.x - f.x,t.y - f.y,t.z - f.z]
    
  
  def getMagnitude(self)-> float:
    return math.pow(math.pow(self.vector[0],2.0)+math.pow(self.vector[1],2.0)+math.pow(self.vector[2],2.0),0.5)
  
  def getElements(self) -> str:
    return ''.join(sorted(str(self.atomFrom.element + self.atomTo.element)))
  
  def getIndexes(self) -> list[int]:
    return [self.atomFrom.index,self.atomTo.index]
  
  def getAngleTo(self, other) -> float:
    v1 = np.array(self.vector) #AI written
    v2 = np.array(other.vector)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return np.arccos(np.clip(cos_angle, -1, 1))
    
  def setBondLength(self, angstrom: float):
    self.changeBondLength(angstrom - self.getMagnitude())
  
  def changeBondLength(self, dAngstrom: float):
    multiplier = (dAngstrom + self.getMagnitude()) / dAngstrom
    for i in range(3):
      self.vector[i] = self.vector[i] * multiplier
    self.atomFrom.startRegen()
    
  def setBondPitch(self, angle: float):
    self.changeBondPitch(angle - self.pitch)
  
  def changeBondPitch(self, dAngle: float):
    self.vector = list(rotate_vector_quaternion(np.array(self.vector), [0, 1, 0], dAngle))
    self.atomFrom.startRegen()
    self.pitch += dAngle
  
  def setBondYaw(self, angle: float):
    self.changeBondYaw(angle - self.yaw)
  
  def changeBondYaw(self, dAngle: float):
    self.vector = list(rotate_vector_quaternion(np.array(self.vector), [0, 0, 1], dAngle))
    self.atomFrom.startRegen()
    self.yaw += dAngle
    
  def resetAngle(self, axis: str):
    if axis == 'x':
      self.vector = [self.getMagnitude(), 0.0, 0.0]
    elif axis == 'y':
      self.vector = [0.0, self.getMagnitude(), 0.0]
    elif axis == 'z':
      self.vector = [0.0, 0.0, self.getMagnitude()]
    self.yaw = 0.0
    self.pitch = 0.0
    #self.atomFrom.startRegen()
  
  def __str__(self) -> str:
    return "bond with vector " +str(self.vector) + " with atoms " + str(self.getIndexes()) + " of type " + str(self.bondType)
  
  def getAimMag(self) -> float:
    el = self.getElements()
    ty = self.bondType
    if el == "CH":
      return 1.08
    elif el == "CC":
      if ty == 1:
        return 1.535
      elif ty == 2:
        return 1.339
      elif ty == 3:
        return 1.203
    elif el == "CN":
      if ty == 1:
        return 1.47
      elif ty == 2:
        return 1.27
      elif ty == 3:
        return 1.15
    elif el == "CO":
      if ty == 1:
        return 1.43
      elif ty == 2:
        return 1.23
      elif ty == 3:
        return 1.13
    elif el == "NN":
      if ty == 1:
        return 1.47
      elif ty == 2:
        return 1.24
      elif ty == 3:
        return 1.10
    elif el == "NO":
      if ty == 1:
        return 1.44
      elif ty == 2:
        return 1.20
    elif el == "OO":
      if ty == 1:
        return 1.48
      elif ty == 2:
        return 1.21
    elif el == "CH":
      return 1.09
    elif el == "HO":
      return 0.97
    elif el == "HN":
      return 1.01
    elif el == "OP":
      if ty == 1:
        return 1.55
      elif ty == 2:
        return 1.62
    return -1.0
    
    


class atom():
    element: str
    index: int
    x: float
    y: float
    z: float
    into: list[bond]
    out: list[bond]
    countClean: int
    connections: list[int]
    
    def __init__(self, xI: float, yI: float, zI: float, eI:str, iI: int):
      self.x = xI
      self.y = yI
      self.z = zI
      self.element = eI
      self.index = iI
      self.into = []
      self.out = []
      self.connections = []
      self.countClean = 0
      
    def shiftXYZ(self, xyz: list[float]):
      self.x += xyz[0]
      self.y += xyz[1]
      self.z += xyz[2]
      for i in self.out:
        for j in range(3):
          i.vector[j] -= xyz[j]
      for i in self.into:
        for j in range(3):
          i.vector[j] += xyz[j]
        i.atomFrom.startRegen()
    
    def generateVectOut(self, bondList: list[list]):
      i = 0
      while i < len(bondList):
        if self in bondList[i][:-1]:
          bondList[i].remove(self)
          nBond = bond(self,bondList[i][0],bondList[i][1])
          self.out.append(nBond)
          self.connections.append(bondList[i][0])
          bondList[i][0].addVectInto(nBond)
          bondList.pop(i)
          i += -1
        i += 1
      for i in self.out:
        i.atomTo.generateVectOut(bondList)
    
    def addVectInto(self, inn: bond):
      self.into.append(inn)
      self.connections.append(inn.atomFrom.index)
    
    def startRegen(self):
      self.countClean = int(random.random()*922337203685477580)
      for i in range(0,len(self.out)):
        self.out[i].atomTo.regenXYZ(self.out[i],self.countClean)
    
    def regenXYZ(self, lastBond, cleanCount):
      if self.countClean == cleanCount:
        return
      self.countClean = cleanCount
      self.x = lastBond.atomFrom.x + lastBond.vector[0]
      self.y = lastBond.atomFrom.y + lastBond.vector[1]
      self.z = lastBond.atomFrom.z + lastBond.vector[2]
      for i in range(0,len(self.out)):
        self.out[i].atomTo.regenXYZ(self.out[i],cleanCount)
    
    def checkAngles(self, min: float):
      for i in self.into:
        for j in self.out:
          if (i.getAngleTo(j) < min):
            return False
      for i in self.out:
        if not(i.atomTo.checkAngles(min)):
          return False
      return True
    
    #def checkBondLengths(self, minDev: float, maxDev: float):
    #  for i in range(len(self.out)):
    #    #print("indexes", self.index, ",", self.out[i].index, ", elements", self.element + self.out[i].element,end=", ")
    #    aimLength = self.out[i].getAimMag()
    #    mag = self.out[i].getMagnitude()
    #    #print("nnn, ", mag, ", ", aimLength)
    #    if (mag > aimLength * maxDev or mag < aimLength * minDev):
    #      return False
    #  for i in self.out:
    #    if not(i.atomTo.checkBondLengths(minDev, maxDev)):
    #      return False
    #  return True
    
    def __str__(self):
      returner: str = "atom " + str(self.index) + " at (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + "). Bonds incoming from "
      for i in self.into:
        returner += str(i.atomFrom.index) + ", "
      returner += "and going to "
      for i in self.out:
        returner += str(i.atomTo.index) + ", "
      return returner
          

class molocule():
  atoms: list[atom]
  bonds: list[bond]
  
  def molToVector(self, mol: str):
    atomList = []
    bondList = []

    for i in mol.split("\n"):
      ja = i.split(" ")
      j = [k for k in ja if k != '']
      if len(j) == 16:
        build = atom(float(j[0]),float(j[1]),float(j[2]),j[3], len(atomList))
        atomList.append(build)
      elif len(j) == 4:
        bondList.append([atomList[int(j[0])-1],atomList[int(j[1])-1],int(j[2])])
        #print(int(j[0])-1, ", ", int(j[1])-1)
        #atoms[int(j[0])-1].connections.append(atoms[int(j[1])-1])
        #atoms[int(j[1])-1].connections.append(atoms[int(j[0])-1])

        #atoms[int(j[0])-1].out.append(atoms[int(j[1])-1])
        #if not(atoms[int(j[0])-1] in atoms[int(j[1])-1].out):
        #  atoms[int(j[1])-1].into.append(atoms[int(j[0])-1])
    atomList[0].generateVectOut(bondList)
    trueBondList = []
    for i in atomList:
      trueBondList += i.out
    self.bonds = trueBondList
    self.atoms = atomList

  def vectorToMol(self) -> str:
    returnBuilder: list[str] = ["","  WebMO",""]
    atomNum = 0
    bondNum = 0
    atomsBuild = []
    bondsBuild = []
    for i in self.atoms:
      atomNum += 1
      xyz = [str(round(i.x,3)),str(round(i.y,3)),str(round(i.z,3))]
      for k in range(len(xyz)):
        toGo = 6 if float(xyz[k]) >= 0 else 7
        while len(xyz[k])<toGo:
          xyz[k] += '0'

      atomsBuild.append(("    " if i.x>=0 else "   ") + xyz[0] + 
                        ("    " if i.y>=0 else "   ") + xyz[1] + 
                        ("    " if i.z>=0 else "   ") + xyz[2] + 
                        " " + i.element + "   0  0  0  0  0  0  0  0  0  0  0  0")
      for j in i.out:
        bondNum += 1
        bB = [str(i.index+1),str(j.atomTo.index+1)]
        while len(bB[0]) < 3:
          bB[0] = " " + bB[0]
        while len(bB[1]) < 3:
          bB[1] = " " + bB[1]
        typ = j.bondType
        bondsBuild.append(bB[0] + bB[1] + "  " + str(typ) + "  0")
    returnBuilder.append(" " + str(atomNum) + " " + str(bondNum) + "  0        0              1 V2000")
    returnBuilder += atomsBuild
    returnBuilder += bondsBuild
    returnBuilder.append("M  END")
    return "\n".join(returnBuilder)
  
  def resetBondAngles(self, axis: str = 'x'):
    for i in range(len(self.bonds)):
      self.bonds[i].resetAngle(axis)
    self.atoms[0].startRegen()
  
  
  
  def scoreValidity(self, minAngle: float = np.pi/8.0, minLengthMult: float = 0.3, maxLengthMult: float = 1.8, minDistanceMult: float = 0.3) -> float:
    score: float = 0.0
    for i in range(len(self.atoms)-1):
      for j in range(i+1,len(self.atoms)-1):
        if not(j in self.atoms[i].connections):          
          testBond = bond(self.atoms[i],self.atoms[j],1)
          if (testBond.getMagnitude() < testBond.getAimMag() * minDistanceMult):
            score += testBond.getAimMag() * minDistanceMult - testBond.getMagnitude()
    for i in self.bonds:
      if i.getMagnitude() > i.getAimMag() * maxLengthMult:
        score += i.getMagnitude() - i.getAimMag() * maxLengthMult
      elif i.getMagnitude() < i.getAimMag() * minLengthMult:
        score += i.getAimMag() * minLengthMult - i.getMagnitude()
    for i in self.atoms:
        for j in i.into:
          for k in i.out:
            if (j.getAngleTo(k) < minAngle):
              score += (minAngle - j.getAngleTo(k)) * np.pi
    return score #and self.atoms[0].checkBondLengths(minLengthMult,maxLengthMult)
  
  
  def __str__(self):
    return "molocule containing " + str(len(self.atoms)) + " atoms and " + str(len(self.bonds)) + " bonds."





def rotate_vector_quaternion(vector, axis, angle: float): #AI written
  #AXIS: first index is roll, second is pitch, third is yaw
    """Rotate using quaternions (more numerically stable)."""
    axis = np.array(axis) / np.linalg.norm(axis)
    # Quaternion components: (w, x, y, z)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    q = np.array([w, *xyz])
    # Vector as quaternion (scalar = 0)
    v_q = np.array([0, *vector])
    # Quaternion rotation: v' = q * v * q^(-1)
    # Using conjugates since quaternions are unit length
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    # Manual quaternion multiplication
    def quat_mult(a, b):
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
        ])
    result = quat_mult(quat_mult(q, v_q), q_conj)
    return result[1:]  # Return vector part