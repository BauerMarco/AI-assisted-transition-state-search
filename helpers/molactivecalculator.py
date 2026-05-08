import numpy as np
import torch
from theforce.calculator.active import ActiveCalculator
from ase.units import GPa


class MolecularActiveCalculator(ActiveCalculator):
    """Subclass of ActiveCalculator with stress and cell gradient disabled.
    
    Suitable for non-periodic molecular systems where stress is meaningless.
    Eliminates:
      - cell gradient computation (saves autograd overhead)
      - stress logging in _exact()
      - get_stress() call on the base calculator
    """

    def grads(self, energy, retain_graph=False):
        """Override to skip cell gradient and return zero stress."""
        if energy.grad_fn:
            forces = -torch.autograd.grad(
                energy,
                self.atoms.xyz,
                retain_graph=retain_graph,
                allow_unused=True,
            )[0]
            if forces is None:
                forces = torch.zeros_like(self.atoms.xyz)
        else:
            forces = torch.zeros_like(self.atoms.xyz)

        if self.atoms.is_distributed:
            import theforce.distributed as distrib
            distrib.all_reduce(forces)

        # Return zero stress — meaningless for molecules
        stress = np.zeros((3, 3))
        return forces, stress

    def _exact(self, copy, _calc=None, task=None):
        """Override to skip get_stress() call and stress logging."""
        from ase.calculators.singlepoint import SinglePointCalculator

        tmp = copy.as_ase() if self.to_ase else copy
        tmp.calc = _calc or self._calc
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        stress = np.zeros(6)  # Voigt, eV/Å³

        if self.tape:
            self._saved_for_tape = tmp

        self.log(f"exact energy: {energy}")
        # stress logging intentionally omitted

        if self.model.ndata > 0:
            if task is None:
                dE = self.results["energy"] - energy
                df = abs(self.results["forces"] - forces)
                self.log("errors (pre):  del-E: {:.2g}  max|del-F|: {:.2g}"
                         "  mean|del-F|: {:.2g}".format(dE, df.max(),
                                                        df.mean()))
            else:
                dE = self.results["energy"][task] - energy
                df = abs(self.results["forces"][..., task] - forces)
                self.log("errors (pre):  del-E: {:.2g}  max|del-F|: {:.2g}"
                         "  mean|del-F|: {:.2g}".format(dE, df.max(),
                                                        df.mean()))

        self._last_test = self.step
        return energy, forces, stress
