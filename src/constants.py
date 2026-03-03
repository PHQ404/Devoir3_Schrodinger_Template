"""
Physical and numerical constants used throughout the package.
"""

import numpy as np

LEFT_BOUNDARY = np.array([0, 0.001])
"""Initial conditions at the left boundary for the shooting method.

The first element is the value of the wavefunction :math:`\\psi(x_0) = 0`
and the second element is its derivative :math:`\\psi'(x_0) = 0.001`.
"""
