# Built-in modules
from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor

# Third party modules
import numpy as np
import pandas as pd
import logging

# Project specific modules
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


def Itx(xc, x_sb, y_sb, yr, Ec=30*10**6, Es=210*10**6):
  '''
  Return moment of inertia about the x-axis of a transformed reinforced concrete cross 
  section. The section is assumed oriented with x as the horizontal axis.

  Args:
    xc (float)            : x-coordinate of elastic centroid
    x_sb (list)           : x-coordiantes of stress block vertices
    y_sb (list)           : y-coordiantes of stress block vertices
    yr (list)             : y-coordiantes of rebars
    Ec (float, optional)  : Young's modulus for concrete (defaults to 30*10**6)
    Es (float, optioanl)  : Young's modulus for reinforcement (defaults to 210*10**6 )

  Returns:
    Itx (float) : Moment of inertia about the x-axis
  '''
  pass


def Ity(yc, x_sb, y_sb, xr, Ec=30*10**6, Es=210*10**6):
  '''
  Return moment of inertia about the y-axis of a transformed reinforced concrete cross 
  section. The section is assumed oriented with y as the vertical axis.

  Args:
    yc (float)            : y-coordinate of elastic centroid
    x_sb (list)           : x-coordinates of stress block vertices
    y_sb (list)           : y-coordiantes of stress block vertices
    xr (list)             : x-coordinates of rebars
    Ec (float, optional)  : Young's modulus for concrete (defaults to 30*10**6)
    Es (float, optioanl)  : Young's modulus for reinforcement (defaults to 210*10**6 )

  Returns:
    Ity (float) : Moment of inertia about the y-axis
  '''
  pass


def elastic_centroid():
  '''
  Return elastic centroid of reinforced concrete sections.

  Args:
    par1 (type) : 
    par2 (type) : 

  Returns:
    ret1 (type) :
  
  TODO

  '''
  pass
