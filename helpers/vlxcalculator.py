"""
veloxchem_ase.py
================
A minimal ASE calculator wrapper for VeloxChem.

Supports closed-shell (restricted) DFT single-point energy and
analytic gradient calculations. Dispersion correction (D4) is
enabled by default.

Requirements
------------
- veloxchem  (conda install -c veloxchem veloxchem)
- ase >= 3.17
- numpy

Units
-----
VeloxChem works in atomic units throughout:
  - Energy    : Hartree  -> eV    (multiply by ase.units.Hartree)
  - Gradient  : Ha/Bohr  -> eV/Å  (multiply by ase.units.Hartree / ase.units.Bohr)
  - Positions : Angstrom (VeloxChem and ASE agree here, no conversion needed)

Usage
-----
    from ase.build import molecule
    from veloxchem_ase import VeloxChemCalculator

    atoms = molecule("H2O")
    atoms.calc = VeloxChemCalculator(xcfun="b3lyp", basis="def2-svp")

    energy = atoms.get_potential_energy()   # eV
    forces = atoms.get_forces()             # eV/Å
"""

import io
from pathlib import Path

import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase import units
from ase import Atoms
from ase.io import write as ase_write
import veloxchem as vlx


class VeloxChemCalculator(Calculator):
    """Minimal ASE interface to VeloxChem for closed-shell DFT.

    Parameters
    ----------
    xcfun : str
        Exchange-correlation functional recognised by VeloxChem / Libxc.
        Default: ``"b3lyp"``
    basis : str
        Basis set name. Default: ``"def2-svp"``
    dispersion : bool
        Enable DFT-D4 dispersion correction. Default: ``True``
    charge : int
        Net molecular charge. Default: ``0``
    multiplicity : int
        Spin multiplicity (2S+1). Must be 1 for closed-shell. Default: ``1``
    warm_start : bool
        Reuse the converged MOs from the previous SCF as the initial guess
        for the next call. Significantly reduces SCF iterations during a
        geometry optimisation or NEB/dimer run. Default: ``True``
    checkpoint_file : str
        Path to the HDF5 checkpoint file used for warm-starting. The file is
        written after each successful SCF and read at the start of the next.
        Default: ``"vlx_ase_guess.h5"``
    mute : bool
        Wether to mute the output streams of the SCF and Gradient driver
    """

    implemented_properties = ["energy", "forces", "stress"]

    default_parameters = {
        "xcfun": "b3lyp",
        "basis": "def2-svp",
        "dispersion": True,
        "charge": 0,
        "multiplicity": 1,
        "warm_start": True,
        "checkpoint_file": "vlx_ase_guess.h5",
        "mute": True,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Tracks whether a valid checkpoint exists from a previous call.
        self._checkpoint_ready = False

    # ------------------------------------------------------------------
    # Public entry point called by ASE optimisers and NEB/dimer methods
    # ------------------------------------------------------------------

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        print("VeloxChemCalculator: Starting calculation with properties:",
              properties,
              flush=True)
        if properties is None:
            properties = self.implemented_properties

        # Mandatory: attach atoms, create scratch directory if needed.
        super().calculate(atoms, properties, system_changes)

        # Guard: VeloxChem is a molecular code; periodic systems not supported.
        if self.atoms.pbc.any():
            raise NotImplementedError(
                "VeloxChemCalculator does not support periodic boundary "
                "conditions. Set atoms.pbc = False.")

        # Guard: only closed-shell (singlet) calculations are supported here.
        if self.parameters.multiplicity != 1:
            raise NotImplementedError(
                "VeloxChemCalculator currently supports only closed-shell "
                "(multiplicity=1) calculations.")

        # ---- Build VeloxChem Molecule and basis objects ----
        molecule = self.ase_to_vlx(self.atoms)
        molecule.set_charge(self.parameters.charge)
        molecule.set_multiplicity(self.parameters.multiplicity)
        basis = vlx.MolecularBasis.read(molecule, self.parameters.basis)

        # ---- Configure SCF driver ----
        scf_drv = vlx.ScfRestrictedDriver()
        if self.parameters.mute:
            scf_drv.ostream.mute()
        scf_drv.xcfun = self.parameters.xcfun
        scf_drv.dispersion = self.parameters.dispersion

        # ---- Warm-start: point driver at checkpoint if one is available ----
        checkpoint_path = Path(self.parameters.checkpoint_file)
        if self.parameters.warm_start and self._checkpoint_ready:
            scf_drv.checkpoint_file = str(checkpoint_path)
            scf_drv.restart = True

        # ---- Run SCF ----
        scf_results = scf_drv.compute(molecule, basis)

        # ---- Stress: zero for a non-periodic molecular system ----
        # AutoForce unconditionally requests stress; returning zeros is
        # physically correct since stress is undefined for molecules.
        self.results["stress"] = np.zeros(6)

        # ---- On success, arm the checkpoint for the next call ----
        if self.parameters.warm_start:
            if scf_drv.is_converged:
                # The driver writes the checkpoint automatically when
                # checkpoint_file is set; ensure it is set so the file is
                # written, then flag that a valid checkpoint now exists.
                scf_drv.checkpoint_file = str(checkpoint_path)
                self._checkpoint_ready = True
            else:
                # Discard a potentially stale checkpoint so the next call
                # falls back to a clean SAD guess rather than a broken one.
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                self._checkpoint_ready = False

        # ---- Extract total energy (Hartree -> eV) ----
        energy_hartree = scf_results["scf_energy"]
        self.results["energy"] = energy_hartree * units.Hartree

        # ---- Compute analytic gradient if forces are requested ----
        if "forces" in properties:
            grad_drv = vlx.ScfGradientDriver(scf_drv)
            grad_drv.numerical = False
            grad_drv.compute(molecule, basis)

            # Gradient shape: (N, 3) in Hartree/Bohr.
            # Forces = -gradient; convert to eV/Å.
            gradient = np.array(grad_drv.gradient)
            conversion = units.Hartree / units.Bohr
            self.results["forces"] = -gradient * conversion

        print(
            f"VeloxChem SCF converged: {scf_drv.is_converged}, energy = {self.results['energy']:.6f} eV",
            flush=True)

    # ------------------------------------------------------------------
    # Helper: convert ASE Atoms -> vlx.Molecule
    # ------------------------------------------------------------------
    @staticmethod
    def ase_to_vlx(ase_atoms):
        """Build a VeloxChem Molecule from the current ASE Atoms object.

        Uses ASE's built-in ``ase.io.write`` to produce a standard XYZ string
        (positions in Angstrom), which VeloxChem's ``Molecule.read_xyz_string``
        accepts directly — no unit conversion needed for coordinates.
        """
        buf = io.StringIO()
        ase_write(buf, ase_atoms, format="xyz")
        xyz_string = buf.getvalue()

        vlx_mol = vlx.Molecule.read_xyz_string(xyz_string)

        return vlx_mol

    @staticmethod
    def vlx_to_ase(vlx_mol):
        """Convert a VeloxChem Molecule object to an ASE Atoms object.

        Transfers atomic symbols, positions (converting from Bohr to Ångström),
        charge, and multiplicity (stored in atoms.info).

        Parameters
        ----------
        molecule : veloxchem.Molecule
            A VeloxChem Molecule object.

        Returns
        -------
        ase.Atoms
            The equivalent ASE Atoms object.
        """

        symbols = vlx_mol.get_labels(
        )  # list of element symbols, e.g. ['O', 'H', 'H']

        # Coordinates are in Bohr; stack into (N, 3) and convert to Ångström
        positions = vlx_mol.get_coordinates_in_angstrom()  # shape (N, 3)

        ase_atoms = Atoms(symbols=symbols, positions=positions)

        # Preserve charge and multiplicity in atoms.info so they aren't lost

        return ase_atoms
