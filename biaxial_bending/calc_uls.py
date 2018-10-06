from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import geometry
import section_calc as sc
import section_plot_uls 

'''
DESCRIPTION

    Sign convention
      Coordinate system           :   left handed
      Tension                     :   Positive
      Compression                 :   Negative

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

# TODO Neutral axis rotation should about plastic centroid, see 'Structural Analysis of Cross Sections', p. 190
# TODO Check for EQ between P, C and T after each run

# Create log file and set default logging statement
FORMAT = '%(name)-15s %(message)s'
filename = 'test.log'
logging.basicConfig(filename=filename, level=logging.DEBUG, filemode='w', format=FORMAT)

np.set_printoptions(precision=2)


def compute_capacities(Fc, Fr, Mcx, Mcy, Mrx, Mry):
    '''    Returns capacities P, Mx and My    '''
    # Total capacities
    P = sum(Fr) + Fc
    Mx = sum(Mrx) + Mcx
    My = sum(Mry) + Mcy

    return P, Mx, My


def compute_capacity_surface(x, y, xr, yr, fcd, fyd, Es, eps_cu, As, lambda_=0.80,  rotation_step=5, vertical_step=10):
    ''' Returns coordinates for capacity surface of cross section (axial load and moments)'''
    # TODO Find a good way to define steps and loop over entire function
    # TODO Find a better way to represent increments for na_y, right now 0 is being computed twice __
    # TODO __ Stop varying na_y if pure tension or compression is found, i.e. if the moment capacities both become 0 __
    # TODO __ See GitHub Issue #2
    n = 8
    na_y_list = [-16*4/2*i/n for i in range(n)] + [16*4/2*i/n for i in range(n)]    # FIXME Improve!
    alpha_list = [alpha for alpha in range(0, 360, 5)]

    P_list = []
    Mx_list = []
    My_list = []
    na_y_computed = []
    alpha_computed = []
    for na_y in na_y_list:
        for alpha_deg in alpha_list:

            # # Perform cross section analysis (NOTE This should maybe be a function itself returning Fc and Fr)
            # dv, dr = sc.compute_dist_from_na_to_vertices(x, y, xr, yr, alpha_deg, na_y)
            # x_sb, y_sb, Asb, sb_cog, c = sc.compute_stress_block_geometry(x, y, dv, dr, alpha_deg, na_y)
            # eps_r = sc.compute_rebar_strain(dr, c, eps_cu)
            # sigma_r = sc.compute_rebar_stress(eps_r, Es, fyd)
            # rebars_inside = sc.get_rebars_in_stress_block(xr, yr, x_sb, y_sb)
            # Fr = sc.compute_rebar_forces(xr, yr, As, sigma_r, rebars_inside, fcd, lambda_=lambda_)
            # Fc = sc.compute_concrete_force(fcd, Asb)
            
            # Perform cross section ULS analysis
            Fc, Fr, Asb, sb_cog, _, _ = sc.perform_section_analysis(x, y, xr, yr, fcd, fyd, Es, eps_cu, As, alpha_deg, na_y, lambda_=0.80)
            
            # Compute individual moments generated in the section
            Mcx, Mcy, Mrx, Mry = sc.compute_moment_contributions(xr, yr, Asb, sb_cog, Fc, Fr)

            # Compute capacities
            P, Mx, My = compute_capacities(Fc, Fr, Mcx, Mcy, Mrx, Mry)

            # Update lists of calculated pairs of vertical local and anlge for neutral axis
            na_y_computed.append(na_y)
            alpha_computed.append(alpha_deg)

            # Store iteration results
            P_list.append(P)
            Mx_list.append(Mx)
            My_list.append(My)

    return P_list, Mx_list, My_list, na_y_computed, alpha_computed


if __name__ == '__main__':

    # Define materials
    # NOTE eps_cu = 0.00035 in Eurocode for concrete strengths < C50
    EPS_CU = 0.003      # Compressive crushing strain of concrete (strain when cross section capacity is reached)
    FCK =  4    # [ksi]
    GAMMA_C = 1.0
    FCD = FCK/GAMMA_C
    ES = 29000  # [ksi]
    FYK = 60     # [ksi]
    GAMMA_S = 1.0
    FYD = FYK/GAMMA_S
    AS = 1  # [in^2]

    # Define concrete geometry by polygon vertices
    x = [-8, 8, 8, -8]
    y = [8, 8, -8, -8]
    # x = [8, 8, -8]
    # y = [8, -8, -8]

    # Define rebar locations and sizes
    xr = [-5.6, 0,   5.6,  5.6,  5.6,  0,   -5.6, -5.6]
    yr = [ 5.6, 5.6, 5.6,  0,   -5.6, -5.6, -5.6,  0]
    # xr = [5.6,  5.6,  5.6,  1.0,   -3.5, 1.0]
    # yr = [3.5,  -1.0,   -5.6, -5.6, -5.6, -1.0]

    # Ø = ['insert rebar sizes']    # IMPLEMENT
    Ø = 1
    # As = pi * (Ø / 2)**2   # [in^2]    FIXME This is only the area of a single bar
    As = 0.79

    # NOTE lambda = 0.8 in Eurocode for concrete strengths < C50
    beta_1 = 0.85      # Factor for compression zone height of Whitney stress block
    LAMBDA = beta_1

    # FIXME ZeroDivisionError in 'eps_r.append(dr[i] / c * eps_cu)' for (alpha, na_y)=(45, -16) or (0, -8) ___
    # FIXME ___ Happens just as the section goes from almost pure compression to pure compression. See plot!

    # FIXME Compression zone is not computed correctly if na is below section, FIX!!! Same problem as above comment I think!
    alpha_deg = 15               # [deg]
    na_y = -2       # [in] Distance from top of section to intersection btw. neutral axis and y-axis
    # NOTE na_y Should be infinite if alpha is 90 or 270

    P, Mx, My, na_y_computed, alpha_computed = compute_capacity_surface(x, y, xr, yr, FCD, FYD, ES, EPS_CU, AS, lambda_=LAMBDA)

    # Plot capacity surface
    # section_plot_uls.plot_capacity_surface(Mx, My, P)

    df = pd.DataFrame({'Mx': Mx, 'My': My, 'P': P, 'na_y': na_y_computed, 'alpha': alpha_computed})
    df.to_csv('df_results.csv', sep='\t')


    # Choose a location of the neutral axis
    alpha_deg = 20
    na_y = -2

    # Compute force for neutral axis location
    Fc, Fr, Asb, sb_cog, x_sb, y_sb = sc.perform_section_analysis(x, y, xr, yr, FCD, FYD, ES, EPS_CU, AS, alpha_deg, na_y, lambda_=LAMBDA)

    # Compute individual moments generated in the section
    Mcx, Mcy, Mrx, Mry = sc.compute_moment_contributions(xr, yr, Asb, sb_cog, Fc, Fr)

    # Compute capacities 
    P, Mx, My = compute_capacities(Fc, Fr, Mcx, Mcy, Mrx, Mry)

    # Plot section for specific location of neutral axis
    section_plot_uls.plot_ULS_section(x, y, xr, yr, x_sb, y_sb, Asb, sb_cog, Fc, Fr, Mcx, Mcy, Mrx, Mry, Mx, My, alpha_deg, na_y)

#####################################################
# LOGGING STATEMENTS
#####################################################
# logging.debug('Cross Section State    ' + cross_section_state)
# logging.debug('na_xint          ' + str(np.around(na_xint, decimals=2)))
# logging.debug('na_yint          ' + str(np.around(na_yint, decimals=2)))
# logging.debug('dv               ' + str(np.around(dv, decimals=2)))
# if cross_section_state == 'MIXED TENSION/COMPRESSION':
#     logging.debug('sb_xint          ' + str(np.around(sb_xint, decimals=2)))
#     logging.debug('sb_yint          ' + str(np.around(sb_yint, decimals=2)))
#     logging.debug('x-intersections btw. sb and section: {}'.format(sb_xint, '%.2f'))
#     logging.debug('y-intersections btw. sb and section: ' + str(sb_yint))
#     logging.debug('x_compr_vertices ' + str(x_compr_vertices))
#     logging.debug('y_compr_vertices ' + str(y_compr_vertices))
# if Asb != 0:
#     logging.debug('x_sb:        ' + str(np.around(x_sb, decimals=2)))
#     logging.debug('y_sb:        ' + str(np.around(y_sb, decimals=2)))
#     logging.debug('Asb:         ' + str(np.around(Asb, decimals=2)))
#     logging.debug('sb_cog       ' + str(np.around(sb_cog, decimals=2)))
# logging.debug('c                ' + str(c))
# logging.debug('Lever arm - dr   ' + str(np.around(dr, decimals=2)))
# logging.debug('Strain - Rebars  ' + str(eps_r))
# logging.debug('Stress - Rebars  ' + str(np.around(sigma_r, decimals=2)))
# logging.debug('Fr               ' + str(np.around(Fr, decimals=2)))
# logging.debug('sum(Fr)        ' + str(np.sum(Fr)))
# logging.debug('Fc               ' + str(np.around(Fc, decimals=2)))
# logging.debug('Mcx              ' + str(np.around(Mcx, decimals=2)))
# logging.debug('Mcy              ' + str(np.around(Mcy, decimals=2)))
# logging.debug('Mrx              ' + str(np.around(Mrx, decimals=2)))
# logging.debug('Mry              ' + str(np.around(Mry, decimals=2)))
# logging.debug('sum(Mrx)         ' + str(np.around(sum(Mrx), decimals=2)))
# logging.debug('sum(Mry)         ' + str(np.around(sum(Mry), decimals=2)))
# logging.debug('(P, Mx, My)      ' + str((np.around(P, decimals=2), np.around(Mx, decimals=2), np.around(My, decimals=2))))
# if phi is not None:
#     logging.debug('phi              ' + str(np.around(phi, decimals=2)))
# logging.debug('C:               {:.1f}'.format(C))
# logging.debug('T:               {:.1f}'.format(T))
# logging.debug('My_compr:        ' + str(My_compr))
# logging.debug('Mx_compr:        ' + str(Mx_compr))
# logging.debug('My_C:            {:.1f}'.format(My_C))
# logging.debug('Mx_C:            {:.1f}'.format(Mx_C))
# logging.debug('ey_C:            {:.1f}'.format(ey_C))
# logging.debug('ex_C:            {:.1f}'.format(ex_C))
# logging.debug('My_T:            {:.1f}'.format(My_T))
# logging.debug('Mx_T:            {:.1f}'.format(Mx_T))
# logging.debug('ey_T:            {:.1f}'.format(ey_T))
# logging.debug('ex_T:            {:.1f}'.format(ex_T))
