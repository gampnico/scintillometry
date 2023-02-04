"""Copyright 2019-2023 Helen Ward, Nicolas Gampierakis, Josef Zink.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

=====

Iterative methods to calculate heat fluxes.

For IterationMost, the original code was a Python port of a method
initially written in R by Dr. Helen Ward, but has since been modified.

The mathematical theory behind this iterative scheme is available in::

    - Scintec AG (2022). Scintec Scintillometers Theory Manual,
    Version 1.05. Scintec AG, Rottenburg, Germany.

    - Scintec AG (2008). Scintec Boundary Layer Scintillometer User
    Manual, Version 1.49. Scintec AG, Rottenburg, Germany.

This method has been updated to resolve some inconsistencies between
the manuals, and has additional modifications to enhance performance and
broaden its use with different scintillometer models. Results may
therefore differ slightly between this and previous iterative methods.

TODO:
    - ST-45: Implement MM5 method (Zhang and Anthes, 1982).
    - ST-46: Implement Li method (Li et al., 2014; 2015).
"""

import mpmath

from scintillometry.backend.constants import AtmosConstants


class IterationMost(AtmosConstants):
    """Classic MOST Iteration.

    A detailed description of the iterative scheme is available in::

    Scintec AG (2022). Scintec Scintillometers Theory Manual. Version
    1.05. Scintec AG, Rottenburg, Germany.

    This iterative method resolves some inconsistencies between
    different scintillometer manuals. Results may therefore differ
    slightly with other third-party implementations.

    Attributes:
        constants (AtmosConstants): Class containing atmospheric
            constants.
    """

    def __init__(self):
        super().__init__()
        mpmath.dps = 30
        mpmath.pretty = True
        self.constants = AtmosConstants()

    def momentum_stability_unstable(self, obukhov, z):
        """Integrated stability function for momentum, unstable.

        Uses Paulson (1970).

        Apply when |LOb| < 0.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, z [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
                |Ψm| [0D].

        .. |LOb| replace:: L :sub:`Ob`
        .. |Ψm| replace:: Ψ :sub:`m`
        """

        x = (1 - 16 * (z / obukhov)) ** (1 / 4)
        psi_momentum = (
            2 * mpmath.ln((1 + x) / 2)
            + mpmath.ln((1 + x**2) / 2)
            - 2 * mpmath.atan(x)
            + mpmath.pi / 2
        )

        return psi_momentum

    def momentum_stability_stable(self, obukhov, z):
        """Integrated stability function for momentum, stable.

        Apply when |LOb| > 0.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, z [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
                |Ψm| [0D].

        .. |LOb| replace:: L :sub:`Ob`
        .. |Ψm| replace:: Ψ :sub:`m`
        """

        psi_momentum = (-5) * z / obukhov

        return mpmath.mpf(psi_momentum)

    def momentum_stability(self, obukhov, z):
        """Integrated stability function for momentum.

        Calculates integrated stability function for momentum under
        stable or unstable conditions. No implementation for neutral
        stability.

        Args:
            obukhov (float): Obukhov length, |LOb| [m].
            z (float): Height, z [m].

        Returns:
            mpmath.mpf: Integrated stability function for momentum,
                |Ψm| [0D].

        .. |LOb| replace:: L :sub:`Ob`
        .. |Ψm| replace:: Ψ :sub:`m`
        """

        if obukhov > 0:
            psi_m = self.momentum_stability_stable(obukhov, z)
        else:
            psi_m = self.momentum_stability_unstable(obukhov, z)

        return psi_m
