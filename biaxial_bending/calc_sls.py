# Built-in modules
from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor

# Third party modules
import numpy as np
import pandas as pd
import logging

# Project specific modules
import geometry
import section_calc as sc

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


# Pseudo code for calculating neutral axis by iteration

def find_na(x, y, xr, yr, N, Mx, My, Ec=30*10**6, Es=210*10**6):
        """
        Return neutral axis of a reinforced concrete section given a loadcase (N, Mx, My).

        The section calculation is purely elastic and thus assumes a linear relationship between
        stress and strain.

        The neutral axis is found by fixed point iteration.
        x_(n+1) = f(x_n)   ,   n = 0,1,2 ...
        I.e, the computed x in iteration i is used as guess for iteration i+1 (if x was unsatisfactory)
        Neutral axis has reference point at specified point (x, y).

        Args:
            x (float)         : x-coordinate of cross section vertices
            y (float)         : y-coordinate of cross section vertices
            xr (float)        : x-coordinate of rebars
            yr (float)        : y-coordinate of rebars
            N (float)         : Axial force (negative in compression)
            Mx (float)        : Moment about x-axis
            My (float)        : Moment about y-axis

        Returns:
            yn (float)        : y-coorindate for intersetion between neutral axis and y-axis
            angle (float)     : Angle in degress between neutral axis and x-axis
        """

        # Setup
        itr = 1
        max_itr = 1000
        tol = 0.001

        # Stiffness ratio
        n = Es / Ec
        
        # Starting guess for y-coordinate of neutral axis intersection with y-axis
        y_guess = 0

        # Starting guess for angle between neutral axis and x-axis
        # Angl of resulting moment vector with x-axis coincides with neutral axis for
        # double symmetric sections. Even though they generally not coincide, it might be
        # a good staring guess 
        angle_guess = atan(My/Mx * 180/pi)  # [deg]

        # error = 0


        # Perform iteration of angle between neutral axis and x-axis
        while angle_error > tol and itr < max_itr:

            while yn_error > tol and itr < max_itr:
                # Compute value for max compression strain within the sction strain field
                ''' eps_c_guess =  sc.strain_field_eval(...) '''     

                # Compute distance from neutral axis to concrete vertices and rebars
                dv, dr = sc.compute_dist_to_na(x, y, xr, yr, angle_guess, yn_guess)

                # Compute geometry of the concrete stress block
                x_sb, y_sb, Asb, sb_cog, c = sc.stress_block_geometry(x, y, dv, dr, alpha_deg, yn)

                # Compute rebar stress

                # Compute transformed area of section
                ''' At = Ac + alpha * As    TODO Write function to calc this'''

                # Find elastic centroid based on guesses location 
                ''' Call to function calculating elastic centroid   TODO Write function to do this'''

                # Compute transformed moment of inertia
                ''' Itx = sc.Itx(x, y, yr) '''
                ''' Ity = sc.Itx(x, y, xr) '''

                # Convert moment to centroid of the transformed section
                ''' Mx_guess = Mx + N * arm * 10**(-3) '''
                ''' My_guess = My + N * arm * 10**(-3) '''

                # 

                # difference btw guessed and computed neutral axis
                yn_error = abs(y_guess - yn)

            if error > tol:
                itr += 1

        return x, y, At, It, Mt



if __name__ == '__main__':
    pass
