"""
Molecular structure representation and manipulation module.

This module provides classes for representing molecules as collections of
atoms and bonds, with support for loading/saving MDL Molfile format,
calculating geometric properties, and scoring molecular validity.
"""

import json
import math
import os
import random
from typing import Optional

import numpy as np


# =============================================================================
# Constants
# =============================================================================

_REGEN_SEED_MAX: int = 922337203685477580
"""Maximum value for regeneration seed (2^63 / 2 approximately)."""

_DEFAULT_OUTPUT_DIR: str = "savedFiles"
"""Default directory for saving molecule files."""

# Lookup table for typical bond lengths (elementPair, bondType) -> length in Angstroms
BOND_LENGTHS: dict[tuple[str, int], float] = {
    ("CH", 1): 1.08,
    ("CC", 1): 1.535,
    ("CC", 2): 1.339,
    ("CC", 3): 1.203,
    ("CN", 1): 1.47,
    ("CN", 2): 1.27,
    ("CN", 3): 1.15,
    ("CO", 1): 1.43,
    ("CO", 2): 1.23,
    ("CO", 3): 1.13,
    ("NN", 1): 1.47,
    ("NN", 2): 1.24,
    ("NN", 3): 1.10,
    ("NO", 1): 1.44,
    ("NO", 2): 1.20,
    ("OO", 1): 1.48,
    ("OO", 2): 1.21,
    ("HO", 1): 0.97,
    ("HN", 1): 1.01,
    ("OP", 1): 1.55,
    ("OP", 2): 1.62,
}


# =============================================================================
# Utility Functions
# =============================================================================

def rotateVectorQuaternion(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a vector around an arbitrary axis using quaternion rotation.

    This method is more numerically stable than rotation matrices for
    small successive rotations.

    Parameters
    ----------
    vector : np.ndarray
        The 3D vector to rotate.
    axis : np.ndarray
        The rotation axis (will be normalized).
    angle : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    axis = axis / np.linalg.norm(axis)

    # Quaternion components: (w, x, y, z)
    w = np.cos(angle / 2)
    xyz = axis * np.sin(angle / 2)
    q = np.array([w, *xyz])

    # Vector as quaternion (scalar = 0)
    vQ = np.array([0, *vector])

    # Quaternion conjugate (for unit quaternions, inverse = conjugate)
    qConj = np.array([q[0], -q[1], -q[2], -q[3]])

    def quatMult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        return np.array([
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
        ])

    result = quatMult(quatMult(q, vQ), qConj)
    return result[1:]  # Return vector part (discard scalar)


# =============================================================================
# Bond Class
# =============================================================================

class Bond:
    """
    Represents a chemical bond between two atoms.

    Attributes
    ----------
    atomFrom : Atom
        The atom at the start of the bond vector.
    atomTo : Atom
        The atom at the end of the bond vector.
    bondType : int
        The bond order (1 = single, 2 = double, 3 = triple).
    vector : list[float]
        The 3D vector from atomFrom to atomTo.
    yaw : float
        Current yaw rotation angle in radians.
    pitch : float
        Current pitch rotation angle in radians.
    """

    def __init__(self, atomFrom: 'Atom', atomTo: 'Atom', bondType: int) -> None:
        """
        Initialize a bond between two atoms.

        Parameters
        ----------
        atomFrom : Atom
            The starting atom of the bond.
        atomTo : Atom
            The ending atom of the bond.
        bondType : int
            The bond order (1, 2, or 3).
        """
        self.atomFrom: 'Atom' = atomFrom
        self.atomTo: 'Atom' = atomTo
        self.bondType: int = bondType
        self.vector: list[float] = [
            atomTo.x - atomFrom.x,
            atomTo.y - atomFrom.y,
            atomTo.z - atomFrom.z
        ]
        self.yaw: float = 0.0
        self.pitch: float = 0.0

    def getMagnitude(self) -> float:
        """
        Calculate the length (magnitude) of the bond vector.

        Returns
        -------
        float
            The Euclidean distance between the two atoms.
        """
        return float(np.linalg.norm(self.vector))

    def getElements(self) -> str:
        """
        Get a canonical representation of the bonded elements.

        Returns
        -------
        str
            The two element symbols concatenated in sorted order
            (e.g., "CH" for carbon-hydrogen bond).
        """
        return ''.join(sorted(self.atomFrom.element + self.atomTo.element))

    def getIndexes(self) -> list[int]:
        """
        Get the indices of the two atoms in this bond.

        Returns
        -------
        list[int]
            [indexOfStartAtom, indexOfEndAtom].
        """
        return [self.atomFrom.index, self.atomTo.index]

    def getAngleTo(self, other: 'Bond') -> float:
        """
        Calculate the angle between this bond and another bond.

        Parameters
        ----------
        other : Bond
            The other bond to measure the angle to.

        Returns
        -------
        float
            The angle in radians between the two bond vectors.
        """
        v1 = np.array(self.vector)
        v2 = np.array(other.vector)
        dotProduct = np.dot(v1, v2)
        magnitudeV1 = np.linalg.norm(v1)
        magnitudeV2 = np.linalg.norm(v2)
        cosAngle = dotProduct / (magnitudeV1 * magnitudeV2)
        return float(np.arccos(np.clip(cosAngle, -1, 1)))

    def setBondLength(self, angstrom: float) -> None:
        """
        Set the bond length to a specific value.

        Parameters
        ----------
        angstrom : float
            Desired bond length in Angstroms.
        """
        self.changeBondLength(angstrom - self.getMagnitude())

    def changeBondLength(self, deltaAngstrom: float) -> None:
        """
        Change the bond length by a delta amount.

        Parameters
        ----------
        deltaAngstrom : float
            Amount to change the bond length by.
        """
        currentMagnitude = self.getMagnitude()
        multiplier = (deltaAngstrom + currentMagnitude) / currentMagnitude
        self.vector = [coord * multiplier for coord in self.vector]
        self.atomFrom.startRegen()

    def setBondPitch(self, angle: float) -> None:
        """
        Set the pitch (rotation around Y-axis) to a specific angle.

        Parameters
        ----------
        angle : float
            Desired pitch angle in radians.
        """
        self.changeBondPitch(angle - self.pitch)

    def changeBondPitch(self, deltaAngle: float) -> None:
        """
        Change the pitch by a delta amount.

        Parameters
        ----------
        deltaAngle : float
            Amount to change the pitch by in radians.
        """
        self.vector = list(rotateVectorQuaternion(
            np.array(self.vector),
            np.array([0, 1, 0]),
            deltaAngle
        ))
        self.atomFrom.startRegen()
        self.pitch += deltaAngle

    def setBondYaw(self, angle: float) -> None:
        """
        Set the yaw (rotation around Z-axis) to a specific angle.

        Parameters
        ----------
        angle : float
            Desired yaw angle in radians.
        """
        self.changeBondYaw(angle - self.yaw)

    def changeBondYaw(self, deltaAngle: float) -> None:
        """
        Change the yaw by a delta amount.

        Parameters
        ----------
        deltaAngle : float
            Amount to change the yaw by in radians.
        """
        self.vector = list(rotateVectorQuaternion(
            np.array(self.vector),
            np.array([0, 0, 1]),
            deltaAngle
        ))
        self.atomFrom.startRegen()
        self.yaw += deltaAngle

    def resetAngle(self, axis: str = 'x') -> None:
        """
        Reset the bond vector to be aligned with the specified axis.

        Parameters
        ----------
        axis : str, optional
            The axis to align with ('x', 'y', or 'z'). Defaults to 'x'.
        """
        magnitude = self.getMagnitude()
        if axis == 'x':
            self.vector = [magnitude, 0.0, 0.0]
        elif axis == 'y':
            self.vector = [0.0, magnitude, 0.0]
        elif axis == 'z':
            self.vector = [0.0, 0.0, magnitude]
        self.yaw = 0.0
        self.pitch = 0.0

    def getAimMagnitude(self) -> float:
        """
        Look up the typical bond length for this bond type.

        Returns
        -------
        float
            The expected bond length in Angstroms, or -1.0 if unknown.
        """
        key = (self.getElements(), self.bondType)
        return BOND_LENGTHS.get(key, -1.0)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"Bond with vector {self.vector} "
            f"connecting atoms {self.getIndexes()} "
            f"of type {self.bondType}"
        )


# =============================================================================
# Atom Class
# =============================================================================

class Atom:
    """
    Represents a single atom in a molecule.

    Attributes
    ----------
    element : str
        The chemical element symbol (e.g., "C", "H", "O").
    index : int
        Zero-based index of this atom in the molecule.
    x, y, z : float
        Cartesian coordinates of the atom.
    into : list[Bond]
        Bonds pointing into (toward) this atom.
    out : list[Bond]
        Bonds pointing out from this atom.
    connections : list[int]
        Indices of atoms connected to this one.
    countClean : int
        Regeneration counter for graph traversal.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        element: str,
        index: int
    ) -> None:
        """
        Initialize an atom with position and element information.

        Parameters
        ----------
        x, y, z : float
            Cartesian coordinates.
        element : str
            Chemical element symbol.
        index : int
            Zero-based index in the molecule.
        """
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.element: str = element
        self.index: int = index
        self.into: list[Bond] = []
        self.out: list[Bond] = []
        self.connections: list[int] = []
        self.countClean: int = 0

    def shiftXYZ(self, delta: list[float]) -> None:
        """
        Translate the atom by a 3D displacement vector.

        Parameters
        ----------
        delta : list[float]
            The [dx, dy, dz] displacement vector.
        """
        self.x += delta[0]
        self.y += delta[1]
        self.z += delta[2]

        # Adjust outgoing bond vectors (opposite direction)
        for bond in self.out:
            for j in range(3):
                bond.vector[j] -= delta[j]

        # Adjust incoming bond vectors (same direction) and regenerate
        for bond in self.into:
            for j in range(3):
                bond.vector[j] += delta[j]
            bond.atomFrom.startRegen()

    def generateVectOut(self, bondList: list[list]) -> None:
        """
        Convert bond definitions into directed bond objects.

        Recursively processes the molecule starting from this atom,
        creating outward bonds and building the molecular graph.

        Parameters
        ----------
        bondList : list[list]
            List of [atom1, atom2, bondType] definitions.
        """
        i = 0
        while i < len(bondList):
            bondDef = bondList[i]
            if self in bondDef[:-1]:
                bondDef.remove(self)
                otherAtom = bondDef[0]
                bondType = bondDef[1]
                newBond = Bond(self, otherAtom, bondType)
                self.out.append(newBond)
                self.connections.append(otherAtom.index)
                otherAtom.addVectInto(newBond)
                bondList.pop(i)
                i -= 1
            i += 1

        for bond in self.out:
            bond.atomTo.generateVectOut(bondList)

    def addVectInto(self, incomingBond: Bond) -> None:
        """
        Register an incoming bond to this atom.

        Parameters
        ----------
        incomingBond : Bond
            The bond pointing into this atom.
        """
        self.into.append(incomingBond)
        self.connections.append(incomingBond.atomFrom.index)

    def startRegen(self) -> None:
        """Initiate coordinate regeneration throughout the molecule."""
        self.countClean = int(random.random() * _REGEN_SEED_MAX)
        for bond in self.out:
            bond.atomTo.regenXYZ(bond, self.countClean)

    def regenXYZ(self, lastBond: Bond, cleanCount: int) -> None:
        """
        Recalculate atom coordinates based on bond vectors.

        Parameters
        ----------
        lastBond : Bond
            The bond leading to this atom.
        cleanCount : int
            Current regeneration iteration counter.
        """
        if self.countClean == cleanCount:
            return

        self.countClean = cleanCount
        self.x = lastBond.atomFrom.x + lastBond.vector[0]
        self.y = lastBond.atomFrom.y + lastBond.vector[1]
        self.z = lastBond.atomFrom.z + lastBond.vector[2]

        for bond in self.out:
            bond.atomTo.regenXYZ(bond, cleanCount)

    def checkAngles(self, minAngle: float) -> bool:
        """
        Check if all bond angles meet the minimum threshold.

        Parameters
        ----------
        minAngle : float
            Minimum acceptable angle in radians.

        Returns
        -------
        bool
            True if all angles are acceptable, False otherwise.
        """
        # Check angles at this atom
        for incoming in self.into:
            for outgoing in self.out:
                if incoming.getAngleTo(outgoing) < minAngle:
                    return False

        # Recursively check angles at neighboring atoms
        for bond in self.out:
            if not bond.atomTo.checkAngles(minAngle):
                return False

        return True

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        incomingIndices = ", ".join(str(b.atomFrom.index) for b in self.into)
        outgoingIndices = ", ".join(str(b.atomTo.index) for b in self.out)
        return (
            f"Atom {self.index} ({self.element}) at ({self.x}, {self.y}, {self.z}). "
            f"Bonds incoming from {incomingIndices} and going to {outgoingIndices}."
        )


# =============================================================================
# Molecule Class
# =============================================================================

class Molecule:
    """
    Represents a complete molecule with atoms and bonds.

    Attributes
    ----------
    atoms : list[Atom]
        All atoms in the molecule.
    bonds : list[Bond]
        All bonds in the molecule.
    lastScore : float
        Score from the most recent validity check.
    fromFile : str
        Source file name or identifier.
    """

    def __init__(self) -> None:
        """Initialize an empty molecule."""
        self.atoms: list[Atom] = []
        self.bonds: list[Bond] = []
        self.lastScore: float = 0.0
        self.fromFile: str = ""

    # -------------------------------------------------------------------------
    # Parsing Methods
    # -------------------------------------------------------------------------

    def parseAtomLine(self, fields: list[str], atomList: list[Atom]) -> None:
        """
        Parse a single atom line from mol file format.

        Parameters
        ----------
        fields : list[str]
            Space-separated fields from the mol file line.
        atomList : list[Atom]
            List to append the parsed atom to.
        """
        x = float(fields[0])
        y = float(fields[1])
        z = float(fields[2])
        element = fields[3]
        atomList.append(Atom(x, y, z, element, len(atomList)))

    def parseBondLine(
        self,
        fields: list[str],
        atomList: list[Atom],
        bondList: list[list]
    ) -> None:
        """
        Parse a single bond line from mol file format.

        Parameters
        ----------
        fields : list[str]
            Space-separated fields from the mol file line.
        atomList : list[Atom]
            List of atoms (for index lookup).
        bondList : list[list]
            List to append the bond definition to.
        """
        atomFrom = atomList[int(fields[0]) - 1]  # 1-based to 0-based
        atomTo = atomList[int(fields[1]) - 1]
        bondType = int(fields[2])
        bondList.append([atomFrom, atomTo, bondType])

    def molToVector(self, mol: str, file: str = "") -> None:
        """
        Convert a mol file string to internal vector representation.

        Parses MDL Molfile format, creating atom and bond objects.

        Parameters
        ----------
        mol : str
            Multi-line string in Molfile format.
        file : str, optional
            Source file identifier (stored for reference).
        """
        self.fromFile = file
        atomList: list[Atom] = []
        bondList: list[list] = []

        for line in mol.split("\n"):
            fields = [f for f in line.split(" ") if f]  # Remove empty strings
            if len(fields) == 16:
                self.parseAtomLine(fields, atomList)
            elif len(fields) == 4:
                self.parseBondLine(fields, atomList, bondList)

        # Build directed bond graph starting from first atom
        atomList[0].generateVectOut(bondList)

        # Flatten all outgoing bonds into single list
        self.atoms = atomList
        self.bonds = [bond for atom in atomList for bond in atom.out]

    # -------------------------------------------------------------------------
    # Output Methods
    # -------------------------------------------------------------------------

    def vectorToMol(self) -> str:
        """
        Convert internal representation to Molfile format.

        Returns
        -------
        str
            Multi-line string in MDL Molfile format.
        """
        builder: list[str] = ["", "  WebMO", ""]
        atomNum = len(self.atoms)
        bondNum = sum(len(atom.out) for atom in self.atoms)

        # Build atom lines
        for atom in self.atoms:
            coords = [round(atom.x, 4), round(atom.y, 4), round(atom.z, 4)]
            formattedCoords = []
            for coord in coords:
                coordStr = f"{coord:.4f}"
                if coord >= 0:
                    formattedCoords.append(f"   {coordStr}")
                else:
                    formattedCoords.append(f"  {coordStr}")

            atomLine = (
                f" {formattedCoords[0]} {formattedCoords[1]} {formattedCoords[2]} "
                f"{atom.element}   0  0  0  0  0  0  0  0  0  0  0  0"
            )
            builder.append(atomLine)

        # Build bond lines
        for atom in self.atoms:
            for bond in atom.out:
                idxFrom = f"{atom.index + 1:>3}"
                idxTo = f"{bond.atomTo.index + 1:>3}"
                bondLine = f"{idxFrom}{idxTo}  {bond.bondType}  0"
                builder.append(bondLine)

        # Header line
        builder.append(f" {atomNum} {bondNum}  0        0              1 V2000")
        builder.insert(3, builder.pop())  # Move header to correct position
        builder.append("M  END")

        return "\n".join(builder)

    # -------------------------------------------------------------------------
    # Modification Methods
    # -------------------------------------------------------------------------

    def resetBondAngles(self, axis: str = 'x') -> None:
        """
        Reset all bond vectors to align with the specified axis.

        Parameters
        ----------
        axis : str, optional
            Axis to align bonds with ('x', 'y', or 'z'). Defaults to 'x'.
        """
        for bond in self.bonds:
            bond.resetAngle(axis)
        self.atoms[0].startRegen()

    # -------------------------------------------------------------------------
    # Validation Methods
    # -------------------------------------------------------------------------

    def scoreValidity(
        self,
        power: float = 2.0,
        minAngle: float = np.pi / 4.0,
        minLengthMult: float = 0.8,
        maxLengthMult: float = 1.2,
        minDistanceMult: float = 1
    ) -> float:
        """
        Calculate a score measuring molecular validity.
    
        Penalizes:
        - Atoms too close together
        - Bond lengths deviating from ideal values
        - Bond angles too sharp
    
        Lower scores indicate better validity.
    
        Parameters
        ----------
        power : float, optional
            Exponent for penalty calculations. Defaults to 2.
        minAngle : float, optional
            Minimum acceptable bond angle in radians. Defaults to π/8.
        minLengthMult : float, optional
            Minimum fraction of ideal bond length. Defaults to 0.3.
        maxLengthMult : float, optional
            Maximum fraction of ideal bond length. Defaults to 1.8.
        minDistanceMult : float, optional
            Minimum fraction of ideal length for non-bonded contacts.
            Defaults to 0.3.
    
        Returns
        -------
        float
            The validity score (lower is better).
        """
        score: float = 0.0
        atoms = self.atoms
        bonds = self.bonds
        n = len(atoms)
        anglePenaltyFactor = np.pi
    
        # Check for atoms too close together (non-bonded)
        for i in range(n - 1):
            atomI = atoms[i]
            connectionsI = atomI.connections  # Now a set - O(1) lookup
            xi, yi, zi = atomI.x, atomI.y, atomI.z
    
            for j in range(i + 1, n):
                if j not in connectionsI:  # O(1) with set
                    atomJ = atoms[j]
                    dx = atomJ.x - xi
                    dy = atomJ.y - yi
                    dz = atomJ.z - zi
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    
                    # Compute ideal distance for non-bonded contact
                    bondIJ = Bond(atomI, atomJ, 1)
                    idealDistance = bondIJ.getAimMagnitude() * minDistanceMult
    
                    if distance < idealDistance:
                        score += (idealDistance - distance) ** power
    
        # Check bond lengths
        for bond in bonds:
            aimMagnitude = bond.getAimMagnitude()
            magnitude = bond.getMagnitude()
    
            if magnitude > aimMagnitude * maxLengthMult:
                score += (magnitude - aimMagnitude * maxLengthMult) ** power
            elif magnitude < aimMagnitude * minLengthMult:
                score += (aimMagnitude * minLengthMult - magnitude) ** power
    
        # Check bond angles
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
                        score += ((minAngle - angle) * anglePenaltyFactor) ** power
    
        self.lastScore = score
        return score


    # -------------------------------------------------------------------------
    # Persistence Methods
    # -------------------------------------------------------------------------

    def saveMol(self, path: str = _DEFAULT_OUTPUT_DIR) -> None:
        """
        Save the molecule to a Molfile and update the index.

        Parameters
        ----------
        path : str, optional
            Directory to save the file. Defaults to "savedFiles".
        """
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

        # Generate filename based on existing files
        existingFiles = os.listdir(path)
        molNumber = len([f for f in existingFiles if f.endswith('.mol')])
        filename = f"{molNumber}.mol"

        # Write molecule file
        filepath = os.path.join(path, filename)
        with open(filepath, 'w') as f:
            f.write(self.vectorToMol())

        # Update index file
        indexPath = "moleculeIndex.json"
        try:
            with open(indexPath, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append({
            "fileName": filename,
            "score": self.lastScore,
            "molFrom": self.fromFile
        })

        with open(indexPath, 'w') as f:
            json.dump(data, f, indent=2)

    def __str__(self) -> str:
        """Return a human-readable summary."""
        return f"Molecule with {len(self.atoms)} atoms and {len(self.bonds)} bonds."
