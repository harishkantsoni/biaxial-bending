# Built-in modules
from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor

# Third party modules
import numpy as np
from scipy.spatial import ConvexHull
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

def find_na(x, y, xr, yr, dia, P, Mx, My, fyd, Ec=30*10**6, Es=200*10**6):
    """
    Return neutral axis of a reinforced concrete section given a loadcase (N, Mx, My).

    The section calculation is purely elastic and thus assumes a linear relationship between
    stress and strain.

    The neutral axis is found by fixed point iteration.
    x_(n+1) = f(x_n)   ,   n = 0,1,2 ...
    I.e, the computed x in iteration i is used as guess for iteration i+1 (if x was unsatisfactory)
    Neutral axis has reference point at specified point (x, y).

    Args:
        x (list)        : x-coordinates of cross section vertices
        y (list)        : y-coordinates of cross section vertices
        xr (list)       : x-coordinates of rebars
        yr (list)       : y-coordinates of rebars
        dia (list)      : Diameters of rebars
        P (float)       : Axial force (negative in compression)
        Mx (float)      : Moment about x-axis
        My (float)      : Moment about y-axis
        fyd (float)     : Design yield stress of rebars

    Returns:
        yn (float)        : y-coorindate for intersetion between neutral axis and y-axis
        angle (float)     : Angle in degress between neutral axis and x-axis
    """

    # Setup
    itr_yn = 0
    itr_alpha = 0
    max_itr = 1
    tol = 0.1

    # Stiffness ratio
    n = Es / Ec

    # Initiate error variables and starting guess
    yn_error = 2 * tol
    yn = (max(y) - min(y)) / 2   # Staring guess (could be improved)
    yn_previous = yn
        
    # Iterate intersection between neutral axis and y-axis until P is satisfactory
    while (abs(yn_error) > tol and itr_yn < max_itr):
        itr_yn += 1

        # FIXME THE CONCRETE VERTICES X AND Y SHOULD BE OF THE STRESS BLOCK, NOT THE
        # ENTIRE SECTION AS IT IS NOW!!!!
        # Compute transformed axial stiffness
        EtAt = sc.transformed_axial_stiffness(x, y, xr, yr, dia, P, Ec=Ec, Es=Es)

        # Compute distances from concrete vertices and rebars to neutral axis
        dv, dr = sc.compute_dist_to_na(x, y, xr, yr, 0, yn)

        # Get coordinates of most compressed concrete fiber
        # If section is only in tension this will be the most tensioned fibre
        xmax = x[dv.index(min(dv))]
        ymax = y[dv.index(min(dv))]

        # Compute elastic centroid
        xel, yel = sc.elastic_centroid(x, y, xr, yr, dia, Ec=Ec, Es=Es)

        # Compute transformed moment of inertia
        Itx = sc.Itx(yel, x, y, xr, yr, dia, Ec=Ec, Es=Es)
        Ity = sc.Ity(xel, x, y, xr, yr, dia, Ec=Ec, Es=Es)

        # Compute value for max compression strain within the section strain field
        # In section is only in tension this will be the min tension strain
        eps_max = sc.strain_field_eval( 
            xmax, ymax, P, Mx, My, Ec, EtAt, Itx, Ity)
        # NOTE Parameter 'E' should be an transformed E (I think).

        # NOTE TEST!!!!!!
        eps_max = 0.0035
        # NOTE TEST!!!!!!

        # Compute geometry of the concrete stress block
        
        # Get list of indices for stress block and extract corresponding x- and y-coordinates
        idx_sb = np.where(np.array(dv) <= 0)[0].tolist()    
        dv_sb = [dv[i] for i in idx_sb]     # Distances from neutral axis to compression vertices
        x_sb = [x[i] for i in idx_sb]       # x-coordinates of stress block vertices
        y_sb = [y[i] for i in idx_sb]       # y-coordinates of stress block vertices
        
        # Add interseciton points btw. neutral axis and section to stress block vertices
        dv_sb += [0, 0]     # Intersections are located direction on neutral axis (dist = 0)
        intersections = geometry.line_polygon_collisions(0, yn, x, y)
        x_sb += [intersections[0][0], intersections[0][1]]
        y_sb += [intersections[1][0], intersections[1][1]]

        # Compute force resultant of stress block (volume of stress block)

        # Stresses and strains at vertices
        eps_sb = [di / min(dv_sb) * eps_max for di in dv_sb]  # Strain at stress block vertices
        sigma_sb = [eps_i * Ec for eps_i in eps_sb]   # Concrete stress at stress block vertices

        # Construct points defining volume of stress block
        x_vol = x_sb * 2
        y_vol = y_sb * 2
        z_vol = [0]*len(x_sb) + sigma_sb

        # Compute force resultant (volume)
        points = np.transpose(np.array([x_vol, y_vol, z_vol]))  # Assemble points to numpy array
        Fc = ConvexHull(points).volume             # Force resultant (volume of convex hul)

        # Compute rebar strain
        eps_r = sc.compute_rebar_strain(dr, min(dv), eps_max)

        # Compute rebar stress
        sigma_r = sc.compute_rebar_stress(eps_r, Es, fyd)
        # NOTE There should be a notification if any rebars yields

        # Compute rebar forces
        Fr_each = [sigma_r[i] * pi*dia[i]**2/4 for i in range(len(sigma_r))]

        Fr = sum(Fr_each)
        print('eps_sb =', eps_sb)
        print('Fc =', Fc)
        print('eps_r =', eps_r)
        print('sigma_r =', sigma_r)
        print('Fr_each =', Fr_each)
        print('Fr =', Fr)
        # Compute total force P
        P_guess = Fc + Fr

        # Error between computed axial load P and externally applied P
        yn_error = P - P_guess

        # Update guess if error is not sufficiently small
        # if abs(yn_error) > tol:
        #     itr_yn += 1

        # Strategy for choosing next guess for neutral axis location:
        # If yn_error is positive, P_guess was lower (i.e. more compressive) than it 
        # should have been => move yn up. 
        # If yn_error is negative, P_guess was larger (i.e. more tensile) than it 
        # should have been => move yn down
        if yn_error >= 0:
            # if itr_yn == 1:
            #     yn += 0.1   # Move 1 unit up as there is lack of previously calculated guees   
            # else:
            #     yn += tol
            yn += tol

        else:
            # if itr_yn == 1:
            #     yn -= 0.1   # Move 1 unit up as there is lack of previously calculated guees   
            # else:
            #     yn -= tol 
            yn = yn - 0.01

        # Update current neutral axis guess for comparison in next iteration
        yn_previous = yn

        print('')
        print('max strain =', eps_max)
        print('Number of iterations:', itr_yn)
        print('P =', P)
        print('P_guess =', P_guess)
        print('yn_error =', yn_error)
        print('yn =', yn)
        print('TRUTH CHECK:', abs(yn_error) > tol and itr_yn < max_itr)


    # # Iterate angle between neutral axis and x-axis until Mx and My are satisfactory
    # while alpha_error > tol and itr_alpha < max_itr:
        
    #     if itr_alpha == 1:
    #         # Starting guess for angle between neutral axis and x-axis
    #         # Angl of resulting moment vector with x-axis coincides with neutral axis for
    #         # double symmetric sections. Even though they generally not coincide, it might be
    #         # a good staring guess 
    #         alpha = atan(My/Mx * 180/pi)  # [deg]



    # return yn, alpha
    return yn


# # Compute stress block centroid

# # NOTE Assumed solution: Only vertices have wieghts. They are set equal to 2/3 times 
# # distance to neutral axis. This will have problems, e.g. in the case where vertices
# # are placed at coordinate 0, as the linear slope to next point will not be caught. 
# # A correct way would be to find the centroid of the 3D shape of the stress block.
# Cxsb = sum([x_sb[i] * 2/3 * dv_sb[i] for i in range(len(dv_sb))]) / sum(dv_sb)
# Cysb = sum([y_sb[i] * 2/3 * dv_sb[i] for i in range(len(dv_sb))]) / sum(dv_sb)



if __name__ == '__main__':

    b = 0.250
    h = 0.500
    dia = [pi*0.020**2/4] * 3      # Rebar diameters [in] (No. 7 bars)
    c = 0.040
    Ec = 33 
    Es = 200      
    fyd = 500/1.15

    x = [0, b, b, 0]
    y = [0, 0, h, h]
    xr = [b/3, b/2, 2/3*b]
    yr = [c, c, c]

    P = -80
    Mx = 91
    My = 0

    print('yn =', find_na(x, y, xr, yr, dia, P, Mx, My, fyd, Ec=Ec, Es=Es))
