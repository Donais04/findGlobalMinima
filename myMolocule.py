import numpy as np
import random, math, os, json, time

from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np
from rdkit import Chem
#from openmmforcefields.generators import GAFFTemplateGenerator
from rdkit.Chem import rdForceFieldHelpers
from r2z import *
from r2z.zmatrix import ZMatrix, pts_to_bond, pts_to_angle, pts_to_dihedral

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
    self.yaw = math.atan2(self.vector[1], self.vector[0])
    self.pitch = math.atan2(self.vector[2], self.getMagnitude())
    
  
  def getMagnitude(self)-> float:
    return float(np.linalg.norm(self.vector))
  
  def getElements(self) -> str:
    return ''.join(sorted(str(self.atomFrom.element + self.atomTo.element)))
  
  def getIndexes(self) -> list[int]:
    return [self.atomFrom.index,self.atomTo.index]
  
  def getAngleTo(self, other) -> float:
    #print("line 38", self, "\n", other)
    v1_u = self.vector / np.linalg.norm(self.vector)
    v2_u = other.vector / np.linalg.norm(other.vector)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
  def setBondLength(self, angstrom: float):
    self.changeBondLength(angstrom - self.getMagnitude())
  
  def changeBondLength(self, dAngstrom: float):
    mag = self.getMagnitude()
    multiplier = (dAngstrom + mag) / mag
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
    elif el == "HH":
      return 0.85
    elif el == "CP":
      return 1.83
    elif el == "HP":
      return 1.42
    elif el == "PP":
      return 2.17
    elif el == "NP":
      return 1.49
    raise TypeError("no element found for " + el)
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
    
    def XYZAsVector(self) -> list[float]:
      return ([self.x, self.y, self.z])
    
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
  lastScore: float
  fromFile: str
  
  def molToMine(self, file: str):
    self.fromFile = file
    mol = open(file).read()
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
    atomList[0].generateVectOut(bondList)
    trueBondList = []
    for i in atomList:
      trueBondList += i.out
    self.bonds = trueBondList
    self.atoms = atomList
    self.zm = ZMatrix(self.toRDKitMol())

  def mineToMol(self) -> str:
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
  
  def vectorToInp(
    self,
    name: str,
    metho: str = "r2SCAN-3c"
    ) -> str:
    builder = "#\n# " + name + "\n#\n%MaxCore 4000\n! " + metho + " AutoAux\n\n\n\n* xyz 0 1\n"
    for i in self.atoms:
      builder += i.element
      if i.x < 0.0:
        builder += " " + str(i.x)
      else:
        builder += "  " + str(i.x)
      if i.y < 0.0:
        builder += " " + str(i.y)
      else:
        builder += "  " + str(i.y)
      if i.z < 0.0:
        builder += " " + str(i.z) + "\n"
      else:
        builder += "  " + str(i.z) + "\n"
    
    return builder + "END"

  def vectorToPDB(self, residueName: str = "UNL", chainId: str = "A") -> str: #AI written
    """
    Convert a molocule to a PDB-format string.

    Parameters
    ----------
    mol : molocule
        The molecule to convert.
    residueName : str, optional
        Three-letter residue name. Defaults to "MOL".
    chainId : str, optional
        Single character chain identifier. Defaults to "A".

    Returns
    -------
    str
        PDB-formatted string.
    """
    lines = []
    lines.append("HEADER    MOLECULE TO PDB CONVERSION")
    lines.append("TITLE     CREATED FROM MOLECULE CLASS")
    lines.append(f"REMARK    {len(self.atoms)} ATOMS, {len(self.bonds)} BONDS")

    atomSerial = 1
    for atom in self.atoms:
        recordType = "ATOM  "
        serial = f"{atomSerial:5d}"
        atomName = f"{atom.element:<4s}"
        altLoc = " "
        resName = f"{residueName:<3s}"
        chain = chainId[0]
        resSeq = f"{1:4d}"
        iCode = " "
        x = f"{atom.x:8.3f}"
        y = f"{atom.y:8.3f}"
        z = f"{atom.z:8.3f}"
        occupancy = f"{1.00:6.2f}"
        tempFactor = f"{0.00:6.2f}"
        element = f"{atom.element:>2s}"
        charge = "  "

        line = (
            f"{recordType}{serial} {atomName}{altLoc}{resName} "
            f"{chain}{resSeq}{iCode}   {x}{y}{z}"
            f"{occupancy}{tempFactor}          {element}{charge}"
        )
        lines.append(line)
        atomSerial += 1

    for atom in self.atoms:
        serial = atom.index + 1
        connectedSerials = []

        for bond in atom.out:
            connectedSerials.append(bond.atomTo.index + 1)

        for bond in atom.into:
            if bond.atomFrom.index + 1 not in connectedSerials:
                connectedSerials.append(bond.atomFrom.index + 1)

        if connectedSerials:
            conectLine = "CONECT" + f"{serial:5d}"
            for connSerial in connectedSerials:
                conectLine += f"{connSerial:5d}"
            lines.append(conectLine)

    lines.append("END\n")
    return "\n".join(lines)
  
  
  def toRDKitMol(self, addHydrogens: bool = False) -> Chem.Mol: #AI written
    """
    Convert a Molecule to an RDKit Mol object.

    Parameters
    ----------
    mol : Molecule
        The molecule to convert.
    addHydrogens : bool, optional
        Whether to add implicit hydrogens. Defaults to False.

    Returns
    -------
    Chem.Mol
        RDKit Mol object with 3D coordinates.
    """
    # Create empty editable molecule
    rdkMol = Chem.RWMol()

    # Track index mapping
    atomIndexMap = {}

    # Add atoms
    for atom in self.atoms:
        rdAtom = Chem.Atom(atom.element)
        atomIndex = rdkMol.AddAtom(rdAtom)
        atomIndexMap[atom.index] = atomIndex

    # Add bonds
    for bond in self.bonds:
        idx1 = atomIndexMap[bond.atomFrom.index]
        idx2 = atomIndexMap[bond.atomTo.index]

        # Convert bond type to RDKit bond type
        if bond.bondType == 1:
            rdBondType = Chem.BondType.SINGLE
        elif bond.bondType == 2:
            rdBondType = Chem.BondType.DOUBLE
        elif bond.bondType == 3:
            rdBondType = Chem.BondType.TRIPLE
        else:
            rdBondType = Chem.BondType.SINGLE

        rdkMol.AddBond(idx1, idx2, rdBondType)

    # Convert to editable molecule
    rdkMol = rdkMol.GetMol()

    # CRITICAL: Sanitize the molecule before accessing atom properties
    try:
        Chem.SanitizeMol(rdkMol)
    except Exception as e:
        print(f"Sanitization warning: {e}")
        # Try to fix common issues
        rdkMol = Chem.RWMol(rdkMol)
        rdkMol.UpdatePropertyCache()
        Chem.SanitizeMol(rdkMol)
        rdkMol = rdkMol.GetMol()

    # Add 3D conformer with coordinates
    conformer = Chem.Conformer(len(self.atoms))
    for atom in self.atoms:
        conformer.SetAtomPosition(atom.index, (atom.x, atom.y, atom.z))
    rdkMol.AddConformer(conformer)

    # Add hydrogens if requested
    if addHydrogens:
        rdkMol = Chem.AddHs(rdkMol)
        # Regenerate coordinates after adding Hs
        rdkMol.GetConformer(0).SetAtomPosition

    return rdkMol


  




  def scoreFull(self, args = {}) -> float:
    score = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(self.toRDKitMol(), maxIters=0)[0][1]
    self.lastScore = score
    return score
    s1 = self.scoreValidity(**args)
    if sum(s1) <= 0.0:
      return rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(self.toRDKitMol(), maxIters=0)[0][1]
    return sum(s1) + 12345678
  
  def scoreVerb(self, args = {}):
    score = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(self.toRDKitMol(), maxIters=0)[0][1]
    self.lastScore = score
    return score
    s1 = self.scoreValidity(**args)
    if sum(s1) <= 0.0:
      return rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(self.toRDKitMol(), maxIters=0)[0][1]
    return s1
  
  def resetBondAngles(self, axis: str = 'x'):
    for i in range(len(self.bonds)):
      self.bonds[i].resetAngle(axis)
    self.atoms[0].startRegen()
  
  def mineToList(self) -> list[float]:
    returner = []
    for i in self.bonds:
      returner.append(i.pitch)
      returner.append(i.yaw)
      returner.append(i.getMagnitude())
    return returner
  
  def mineToListXYZ(self) -> list[float]:
    returner = []
    for i in self.bonds:
      returner.append(i.vector[0])
      returner.append(i.vector[1])
      returner.append(i.vector[2])
    return returner
  
  def mineToVectorsXYZ(self) -> list[tuple[float,float,float]]:
    returner = []
    for i in self.bonds:
      returner.append((i.vector[0],i.vector[1],i.vector[2]))
    return returner
  
  def vectorsToMineXYZ(self, l: list[tuple[float,float,float]]) -> None:
    count = 0
    for i in self.bonds:
      i.setBondPitch(l[count][0])
      i.setBondYaw(l[count][1])
      i.setBondLength(l[count][2])
      count += 1
    self.atoms[0].startRegen()
      
  
  def listToMine(self, l: list[float]) -> None:
    count = 0
    for i in self.bonds:
      i.setBondPitch(l[count])
      i.setBondYaw(l[count+1])
      i.setBondLength(l[count+2])
      count += 3
    self.atoms[0].startRegen()
      
  def listToMineXYZ(self, l: list[float]) -> None:
    count = 0
    for i in self.bonds:
      i.vector[0] = l[count]
      i.vector[1] = l[count+1]
      i.vector[2] = l[count+2]
      count += 3
    self.atoms[0].startRegen()
  
  def rando(self) -> None:
    self.resetBondAngles()
    for i in self.bonds:
      i.setBondPitch((random.random() - 0.5) * 2 * np.pi * 2)
      i.setBondYaw((random.random() - 0.5) * 2 * np.pi * 2)
    self.atoms[0].startRegen()
    
  
  def scoreValidity(
    self,
    power: float = 2.0,
    minAngle: float = np.pi / 8.0,
    minLengthMult: float = 0.6,
    maxLengthMult: float = 1.6,
    minDistanceMult: float = 0.8) -> list[float]:
    
    score: list[float] = [0.0, 0.0, 0.0, 0.0]
    atoms = self.atoms
    bonds = self.bonds
    n = len(atoms)
    anglePenaltyFactor = np.pi

    # Check for atoms too close together (non-bonded)
    for i in range(n - 1):
        atomI = atoms[i]
        connectionsI = atomI.connections 
        xi, yi, zi = atomI.x, atomI.y, atomI.z

        for j in range(i + 1, n):
            if j not in connectionsI:
                atomJ = atoms[j]
                dx = atomJ.x - xi
                dy = atomJ.y - yi
                dz = atomJ.z - zi
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                # Compute ideal distance for non-bonded contact
                bondIJ = bond(atomI, atomJ, 1)
                idealDistance = bondIJ.getAimMag() * minDistanceMult

                if distance < idealDistance:
                    score[0] += (idealDistance - distance) ** power

    for bonda in bonds:
        aimMagnitude = bonda.getAimMag()
        magnitude = bonda.getMagnitude()

        if magnitude > aimMagnitude * maxLengthMult:
            score[1] += (magnitude - aimMagnitude * maxLengthMult) ** power
        elif magnitude < aimMagnitude * minLengthMult:
            score[2] += (aimMagnitude * minLengthMult - magnitude) ** power

    for atom in atoms:
        atomInto = atom.into
        atomOut = atom.out

        for incoming in atomInto:
            incomingVector = np.array(incoming.vector)

            for outgoing in atomOut:
                outgoingVector = np.array(outgoing.vector)
                dotProduct = np.dot(incomingVector, outgoingVector)
                magnitudeProduct = np.linalg.norm(incomingVector) * np.linalg.norm(outgoingVector)
                cosAngle = dotProduct / magnitudeProduct
                angle = np.arccos(np.clip(cosAngle, -1, 1))

                if angle < minAngle:
                    score[3] += ((minAngle - angle) * anglePenaltyFactor) ** power

    self.lastScore = sum(score)
    return score

  
  def saveMol(self, filename: str = "", path: str = "savedFiles", fileType: str = "mol", other: list[list] = []):
    if filename == "":
      name = str(int(round(time.time(),3)*1000 % 100000)) + "." + fileType
    else:
      name = filename
    writer = ""
    if fileType == "mol":
      writer = self.mineToMol()
    elif fileType == "xyz":
      writer = self.vectorToInp(name[:-4])
    elif fileType == "pdb":
      writer = self.vectorToPDB()
    if (os.path.exists(path + "/" + name)):
      name = name + "DupeSomehow" #should not happen
    with open(path + "/" + name, 'w') as g:
      g.write(writer)
    data = ""
    with open("moleculeIndex.json", 'r') as f:
      data = json.loads(f.read())
      placer = {
        "fileName":name,
        "score":self.lastScore,
        "molFrom":self.fromFile
      }
      for i in range(len(other)):
        placer[other[i][0]] = other[i][1]
      data.append(placer)
      
    with open("moleculeIndex.json", 'w') as f:
      f.write(json.dumps(data))
      
  
  
  def __str__(self):
    return "molocule containing " + str(len(self.atoms)) + " atoms and " + str(len(self.bonds)) + " bonds."
  
  
  def mineToZMatrix(self):
    
    
    returner = []
    big = self.zm.build_z_crds(np.array(self.toRDKitMol().GetConformers()[0].GetPositions())*ureg.angstrom)
    #print(list(big.values()))
    for i in range(len(big.values())):
      if not (len(str(list(big.values())[i][0]).split(" ")) > 3):
        for j in range(len(list(big.values())[i])):
          temp = str(list(big.values())[i][j]).split(" ")
          value = float(temp[0])
          if temp[1] == "nanometer":
            value = value * 10
          elif temp[1] == "degree":
            value = float(np.deg2rad(value))
          returner.append(value)
          big[i][j] = value
    big[0][0] = [0.0,0.0,0.0]
    #print(big, "\n\n") #toSHOW
    #print("hereeee", self.zm.z)
    with open("output.json", 'w') as f:
      f.write(json.dumps(big))
    
      
    return returner
  
  def getBond(self, a: int, b: int) -> bond:
    for j in range(len(self.bonds)):
      i = self.bonds[j]
      if ((i.atomTo.index == a and i.atomFrom.index == b) or
          (i.atomTo.index == b and i.atomFrom.index == a)):
        return i
    else:
      raise ValueError("bond not found")
    
  def zMatrixToMine(self, zmat):
    #print(self.zm.z)

    
    #print("zmat",zmat)
      
    for i in range(len(self.zm.z)):

        if i == 0:
            self.atoms[0].x, self.atoms[0].y, self.atoms[0].z = 0.0, 0.0, 0.0

        elif i == 1:
            #print(self.getBond(3,2).getMagnitude(), self.zm.z[i], 0)
            b = self.getBond(self.zm.z[i][0], self.zm.z[i][1])
            b.setBondLength(zmat[0])
            #print(self.getBond(3,2).getMagnitude(), self.zm.z[i], 1)

        elif i == 2:
            #print(self.getBond(3,2).getMagnitude(), self.zm.z[i], 0)
            b1 = self.getBond(self.zm.z[i][0],self.zm.z[i][1])
            b2 = self.getBond(self.zm.z[i][2],self.zm.z[i][1])
            #currAngle = angle33DPoints(self.atoms[self.zm.z[i][0]].XYZAsVector(),
            #                            self.atoms[self.zm.z[i][1]].XYZAsVector(),
            #                            self.atoms[self.zm.z[i][2]].XYZAsVector())
            b1.setBondLength(zmat[1])
            self.setAngle(self.atoms[self.zm.z[i][2]],self.atoms[self.zm.z[i][1]],self.atoms[self.zm.z[i][0]],zmat[2])
            #self.rotateBondAroundOther(self.zm.z[i][0], b2, b1, zmat[2] - currAngle)
            #print(self.getBond(3,2).getMagnitude(), self.zm.z[i], 2)

        else:
            zind = (i-2)*3
            b1 = self.getBond(self.zm.z[i][0],self.zm.z[i][1])
            b2 = self.getBond(self.zm.z[i][2],self.zm.z[i][1])
            #currAngle = angle33DPoints(self.atoms[self.zm.z[i][0]].XYZAsVector(),
            #                            self.atoms[self.zm.z[i][1]].XYZAsVector(),
            #                            self.atoms[self.zm.z[i][2]].XYZAsVector())
            #print(zind, self.getBond(30,13).getMagnitude(), self.zm.z[i], 0)
            b1.setBondLength(zmat[zind])
            #print(zind, self.getBond(30,13).getMagnitude(), self.zm.z[i], 1)
            self.setAngle(self.atoms[self.zm.z[i][2]],self.atoms[self.zm.z[i][1]],self.atoms[self.zm.z[i][0]],zmat[zind+1])
            #self.rotateBondAroundOther(self.zm.z[i][0], b2, b1, zmat[zind+1] - currAngle)
            #print(zind, self.getBond(30,13).getMagnitude(), self.zm.z[i], 2)
            
            self.setDihedralAngle(self.atoms[self.zm.z[i][3]],
                                  self.atoms[self.zm.z[i][2]],
                                  self.atoms[self.zm.z[i][1]],
                                  self.atoms[self.zm.z[i][0]],
                                  zmat[zind+2])
            #print(zind, self.getBond(3,2).getMagnitude(), self.zm.z[i], 3)
          
    self.atoms[0].startRegen()
    
  def zmatBounds(self) -> list[tuple[float, float]]:
    bounds = []
    
    # Use the same BFS order as mineToZMatrix
    visited = set()
    order = []
    queue = [self.atoms[0]]
    visited.add(self.atoms[0].index)

    while queue:
        a = queue.pop(0)
        order.append(a)
        neighbors = [b.atomTo for b in a.out] + [b.atomFrom for b in a.into]
        for n in neighbors:
            if n.index not in visited:
                visited.add(n.index)
                queue.append(n)

    # Helper: find bond between two atoms
    def find_bond(a, b):
        for bond in self.bonds:
            if (bond.atomFrom == a and bond.atomTo == b) or \
               (bond.atomFrom == b and bond.atomTo == a):
                return bond
        return None

    for i, atom in enumerate(order):
        if i == 0:
            continue  # No parameters
        
        elif i == 1:
            # r: distance from atom to order[0]
            b = find_bond(atom, order[0])
            if b:
                ideal = b.getAimMag()
                bounds.append((0.6 * ideal, 1.6 * ideal))
            else:
                bounds.append((0.5, 3.0))

        elif i == 2:
            # r: distance from atom to order[1]
            b = find_bond(atom, order[1])
            if b:
                ideal = b.getAimMag()
                bounds.append((0.6 * ideal, 1.6 * ideal))
            else:
                bounds.append((0.5, 3.0))
            # θ: angle at order[1] with order[0]
            try:
                parentBond = bond(atom, order[1], 1)
                grandparentBond = bond(order[1], order[0], 1)
                angle = parentBond.getAngleTo(grandparentBond)
                bounds.append((max(0.1, angle - np.pi/3), min(np.pi - 0.1, angle + np.pi/3)))
            except:
                bounds.append((0.1, np.pi - 0.1))

        else:
            parent = order[i - 1]
            grandparent = order[i - 2]
            greatGrandparent = order[i - 3]
            
            # r: distance from atom to parent
            b = find_bond(atom, parent)
            if b:
                ideal = b.getAimMag()
                bounds.append((0.6 * ideal, 1.6 * ideal))
            else:
                bounds.append((0.5, 3.0))
            
            # θ: angle at parent with grandparent
            try:
                parentBond = bond(atom, parent, 1)
                grandparentBond = bond(parent, grandparent, 1)
                angle = parentBond.getAngleTo(grandparentBond)
                bounds.append((max(0.1, angle - np.pi/3), min(np.pi - 0.1, angle + np.pi/3)))
            except:
                bounds.append((0.1, np.pi - 0.1))
            
            # φ: dihedral defined by greatGrandparent-grandparent-parent-atom
            try:
                dihed = compute_dihedral(
                    [greatGrandparent.x, greatGrandparent.y, greatGrandparent.z],
                    [grandparent.x, grandparent.y, grandparent.z],
                    [parent.x, parent.y, parent.z],
                    [atom.x, atom.y, atom.z]
                )
                bounds.append((dihed - np.pi, dihed + np.pi))
            except:
                bounds.append((-np.pi, np.pi))

    return bounds
  
  def setAngle(self, aIn: atom, bIn: atom, cIn: atom, target_angle: float) -> None: #AI
    a = np.array(aIn.XYZAsVector(), dtype=float)
    b = np.array(bIn.XYZAsVector(), dtype=float)
    c = np.array(cIn.XYZAsVector(), dtype=float)
    ba = a - b
    bc = c - b

    # Normalize
    ba_unit = ba / np.linalg.norm(ba)
    bc_unit = bc / np.linalg.norm(bc)

    # Current angle
    current_angle = angle33DPoints(a,b,c)

    # Rotation needed
    delta_angle = target_angle - current_angle

    # Rotation axis (normal to plane)
    axis = np.cross(ba_unit, bc_unit)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        raise ValueError("Points are colinear; rotation axis is undefined.")

    axis = axis / axis_norm

    # Rodrigues' rotation formula
    def rodrigues(v, k, theta):
        return (v * np.cos(theta) +
                np.cross(k, v) * np.sin(theta) +
                k * np.dot(k, v) * (1 - np.cos(theta)))

    # Rotate bc
    bc_rot = rodrigues(bc, axis, delta_angle)

    # Translate back
    c1 = b + bc_rot
    self.getBond(bIn.index,cIn.index).vector = [c1[0]-bIn.x,c1[1]-bIn.y,c1[2]-bIn.z]
    bIn.startRegen()
    
  def setDihedralAngle(self, aIn: atom, bIn: atom, cIn: atom, dIn: atom, target_angle: float) -> None: #AI
    a, b, c, d = map(lambda x: np.array(x, dtype=float), (aIn.XYZAsVector(), bIn.XYZAsVector(), cIn.XYZAsVector(), dIn.XYZAsVector()))

    # Current dihedral
    current = calculateDihedral(aIn, bIn, cIn, dIn)
    delta = target_angle - current

    # محور rotation = bc axis
    axis = c - b
    axis = axis / np.linalg.norm(axis)

    # Translate so c is origin
    d_rel = d - c

    # Rodrigues rotation
    def rodrigues(v, k, theta):
        return (v * np.cos(theta) +
                np.cross(k, v) * np.sin(theta) +
                k * np.dot(k, v) * (1 - np.cos(theta)))

    d_rot = rodrigues(d_rel, axis, delta)
    d1 = c + d_rot
    self.getBond(cIn.index,dIn.index).vector = [d1[0]-cIn.x,d1[1]-cIn.y,d1[2]-cIn.z]
    bIn.startRegen()
  
  def rotateBondAroundOther(self, center_atom: atom, reference_bond: bond, 
                           target_bond: bond, angle_diff: float) -> None:
    """Rotate target_bond around center_atom by angle_diff radians."""
    ref_vec = np.array(reference_bond.vector)
    ref_norm = ref_vec / np.linalg.norm(ref_vec)
    
    target_vec = np.array(target_bond.vector)
    parallel = np.dot(target_vec, ref_norm) * ref_norm
    perp = target_vec - parallel
    
    if np.linalg.norm(perp) < 1e-10:
        return
    
    perp_norm = perp / np.linalg.norm(perp)
    rot_axis = np.cross(ref_norm, perp_norm)
    
    if np.linalg.norm(rot_axis) < 1e-10:
        return
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    
    cos_a = np.cos(angle_diff)
    sin_a = np.sin(angle_diff)
    K = np.array([[0, -rot_axis[2], rot_axis[1]],
                 [rot_axis[2], 0, -rot_axis[0]],
                 [-rot_axis[1], rot_axis[0], 0]])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    
    new_perp = R @ perp
    new_vec = parallel + new_perp
    
    target_bond.vector = new_vec.tolist()
    target_bond.atomFrom.startRegen()





def calculateDihedral(a: atom, b: atom, c: atom, d: atom) -> float:
    p0 = np.array(a.XYZAsVector())
    p1 = np.array(b.XYZAsVector())
    p2 = np.array(c.XYZAsVector())
    p3 = np.array(d.XYZAsVector())

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    return np.arctan2(y, x)

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



  
  






def angle33DPoints(a1,b1,c1):
  a = np.array(a1)
  b = np.array(b1)
  c = np.array(c1)
  ba = a - b
  bc = c - b

  cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  return np.arccos(cosine_angle)

  


def compute_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def compute_dihedral(p0, p1, p2, p3):
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.arctan2(y, x)
def distance(a, b):
    return np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z])


def angle(a, b, c):
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    return compute_angle(v1, v2)


def dihedral(a, b, c, d):
    return compute_dihedral(
        [a.x, a.y, a.z],
        [b.x, b.y, b.z],
        [c.x, c.y, c.z],
        [d.x, d.y, d.z]
    )


