from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import geometry

'''
DESCRIPTION

    Sign convention
      Coordinate system           :   Left handed
      Tension                     :   Positive
      Compression                 :   Negative

    For problems with biaxial bending and axial load in SLS, the neutral axis can not be computed analytically and
    therefore has to be determined by iteration.
    The iteration procedure guesses a position of the neutral axis and checks for equilibrium between external and
    internal forces.

    The algorithm can look like this:




    - The term 'stress block' is used for the concrete area that, in the calculations, is assumed to be under compression.
      This area depends on the adopted stress-strain curve of the concrete, where the most used one is the Whitney Stress
      Block.
      In reality, everything on one side of the neutral axis is compression.

    -------------- FIND INTERSECTIONS BTW. NEUTRAL AXIS AND SECTION BOUNDARIES -----------
       Equation representing neutral axis:                     0 = tan(alpha) * x - y + na_y
       Equation representing inner edge of strees block        0 = tan(alpha) * x - y + na_y - delta_v
    where
      - 'delta_v' is the vertical distance between the neutral axis and the inner stress block edge
      - 'na_y' is the y-ccordinate of the intersection btw. the y-axis and the neutral axis.
'''
