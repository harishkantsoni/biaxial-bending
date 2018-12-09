# Built-in packages
from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor
import logging

# Third party packages
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Project specific packages
import section_calc as sc
import section_plot_uls
from geometry import point_to_point_dist_3d
from geometry import line_hull_intersection

'''
DESCRIPTION

    Sign convention
      Positive x- and y-axis points to the right and upwards, respectively. Positive moments Mx cause compression at
      positive y-coordinates, while positive moments My cause compression at positve x-coordinates.

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
    vs = vertical_step
    h = max(x) - min(x)
    na_y_list = list(np.linspace((min(x)-h/3), 0, vs)) + list(np.linspace(0, (max(x)+h/3), vs))
    alpha_list = [alpha for alpha in range(0, 360, rotation_step)]

    P_list = []
    Mx_list = []
    My_list = []
    na_y_computed = []
    alpha_computed = []
    for na_y in na_y_list:
        for alpha_deg in alpha_list:

            # Perform cross section ULS analysis
            Fc, Fr, Asb, sb_cog, _, _ = sc.perform_section_analysis(x, y, xr, yr, fcd, fyd, Es, eps_cu, As, alpha_deg, na_y, lambda_=lambda_)

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


from scipy.optimize import linprog


def in_point_cloud(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def utilization_ratio(Ped, Mxed, Myed, P_capsurf, Mx_capsurf, My_capsurf):
    '''
    Return the utilization ratio as the ratio between the distance from the load
    combination point to Origo and the distance from Origo to capacity surface in
    dirrection given by the point.

    Args:
    all inputs are lists...

    Returns
        ur as list
    '''

    # Compute convex hull of the capacity surface point cloud
    cap_surf = np.transpose(np.array([P_capsurf, Mx_capsurf, My_capsurf]))
    convex_hull = ConvexHull(cap_surf)

    # Compute distance from Origo to all load combination points
    comb_array = np.transpose(np.array([Ped, Mxed, Myed]))
    lc_dist = [point_to_point_dist_3d([0, 0, 0], lc) for lc in comb_array]

    # Compute distance from Origo to all capacity surface points in line with load combinations
    # cap_intersections = [line_hull_intersection(lc, convex_hull) for lc in comb_array]
    cap_intersections = [line_hull_intersection(lc, convex_hull) for lc in comb_array]
    cap_dist = [point_to_point_dist_3d([0, 0, 0], cap) for cap in cap_intersections]

    # A for loop is made here in favor of a list comprehension in order to catch
    # the case where all loads in combination is 0. In that case, the Vector
    # has no direction.
    # Compute utilization ratios for all load combinations
    ur = []
    for i in range(len(comb_array)):
        # if not np.all(np.nonzero(comb_array[i])):
        if np.all(comb_array[i] == 0):
            # All external loads are 0, i.e. P=0, Mx=0 and My=0
            ur.append(0.00)
        else:
            # Some loads are nonzero
            ur.append(lc_dist[i] / cap_dist[i])

    return ur

if __name__ == '__main__':

    '''
    1 ksi       ===>  6.895 MPa       (4 ksi          ===>    27.57 MPa)
    1 kip       ===>  4.448 kN        (400 kip        ===>    1779 kN                 1300 kip    ===>    5783 kN)
    1 kip-in.   ===>  0.113 kNm       (3500 kip-in.   ===>    395 kNm)
    '''

    # Define materials

    # EPS_CU = 0.003      # Compressive crushing strain of concrete (strain when cross section capacity is reached)
    # FCK =  4    # [ksi]
    # GAMMA_C = 1.0
    # FCD = FCK/GAMMA_C
    # ES = 29000  # [ksi]
    # FYK = 60     # [ksi]
    # GAMMA_S = 1.0
    # FYD = FYK/GAMMA_S
    # AS = 0.79  # [in^2]

    # Compressive crushing strain of concrete (strain when cross section capacity is reached)
    EPS_CU = 0.0035       # NOTE eps_cu = 0.0035 in Eurocode for concrete strengths < C50
    FCK = 25    # [MPa]
    GAMMA_C = 1.5
    FCD = FCK/GAMMA_C
    ES = 200*10**3  # [MPa]
    FYK = 500     # [MPa]
    GAMMA_S = 1.15
    FYD = FYK/GAMMA_S
    AS = pi*25**2/4  # [mm^2]

    # Define concrete geometry by polygon vertices
    x = [-8, 8, 8, -8]
    y = [8, 8, -8, -8]
    # Convert from [in] to [m]
    x = [i*25.4 for i in x]
    y = [i*25.4 for i in y]

    # Cross seciton shaped like a cross
    x = [-400, 400, 400, 100, 100, 400, 400, -400, -400, -100, -100, -400]
    y = [400, 400, 200, 200, -200, -200, -400, -400, -200, -200, 200, 200]

    # # T-beam
    # x = [-150,-400,-400,400,400,150,150,-150]
    # y = [200,200,400,400,200,200,-150,-150]
    #
    # Triangle shape
    x = [0,-200,200]
    y = [200,-200,-200]

    # x = [8, 8, -8]
    # y = [8, -8, -8]
    # x = [-10, -10, -5, 5,  10, 10,  5,  -5]
    # y = [ -5,  5,  10, 10, 5,  -5, -10, -10]
    # x = [-254, -254, -127, 127, 254, 254, 127, -127]
    # y = [-127, 127, 254, 254, 127, -127, -254, -254]
    # x = [-200, 200, 200, -200]
    # y = [200, 200, -200, -200]

    # Define rebar locations (NOTE Need to be ordered, which should be done automatically)
    xr = [-5.6, 0,   5.6,  5.6,  5.6,  0,   -5.6, -5.6]
    yr = [ 5.6, 5.6, 5.6,  0,   -5.6, -5.6, -5.6,  0]
    # Convert from [in] to [m]
    xr = [i*25.4 for i in xr]
    yr = [i*25.4 for i in yr]

    # Cross shaped section
    xr = [-340, 0,  340, 0, -340, 340, 0]
    yr = [340, 340, 340, 0, -340,-340,-340]

    # # T-beam
    # xr=[-350,-350,-175,175,350,0,350,-100,0,100]
    # yr=[350,250,350,350,350,-100,250,-100,350,-100]
    #
    # Triangle shape
    xr = [-130,130,65,-65,0,0]
    yr = [-160,-160,0,0,140,-160]

    # xr = [5.6,  5.6,  5.6,  1.0,   -3.5, 1.0]
    # yr = [3.5,  -1.0,   -5.6, -5.6, -5.6, -1.0]
    # xr = [-8, -7.8, -4.5,  0,   4.5,  7.8,  8,   7.8,   4.5,   0,    -4.5, -7.8]
    # yr = [ 0,  4.5,  7.8,  8,  7.8,  4.5,   0,   -4.5,  -7.8, -7.8, -7.8, -4.5 ]


    # Ø = ['insert rebar sizes']    # IMPLEMENT
    # Ø = 25
    # As = pi * (Ø / 2)**2   # [in^2]    FIXME This is only the area of a single bar

    # NOTE lambda = 0.8 in Eurocode for concrete strengths < C50
    beta_1 = 0.80      # Factor for compression zone height of Whitney stress block
    LAMBDA = beta_1

    # FIXME ZeroDivisionError in 'eps_r.append(dr[i] / c * eps_cu)' for (alpha, na_y)=(45, -16) or (0, -8) ___
    # FIXME ___ Happens just as the section goes from almost pure compression to pure compression. See plot!

    # FIXME Compression zone is not computed correctly if na is below section, FIX!!! Same problem as above comment I think!
    alpha_deg = 15               # [deg]
    na_y = -50       # Distance from x-axis to intersection btw. neutral axis and y-axis
    # NOTE na_y Should be infinite if alpha is 90 or 270

    # P, Mx, My, na_y_computed, alpha_computed = compute_capacity_surface(x, y, xr, yr, FCD, FYD, ES, EPS_CU, AS, lambda_=LAMBDA)

    # Plot capacity surface
    plot_capacity_surface = 'No'
    if plot_capacity_surface == 'Yes':
        section_plot_uls.plot_capacity_surface(Mx, My, P, plot_type='scatter')

        df = pd.DataFrame({'Mx': Mx, 'My': My, 'P': P, 'na_y': na_y_computed, 'alpha': alpha_computed})
        df.to_csv('df_results.csv', sep='\t')

    # Compute force for neutral axis location
    Fc, Fr, Asb, sb_cog, x_sb, y_sb = sc.perform_section_analysis(x, y, xr, yr, FCD, FYD, ES, EPS_CU, AS, alpha_deg, na_y, lambda_=LAMBDA)

    # Compute individual moments generated in the section
    Mcx, Mcy, Mrx, Mry = sc.compute_moment_contributions(xr, yr, Asb, sb_cog, Fc, Fr)
    print('Mcx = ', Mcx/10**6)
    print('Mcy = ', Mcy/10**6)
    print('Mrx = ', sum(Mrx)/10**6)
    print('Mry = ', sum(Mry)/10**6)
    # Compute capacities
    P_float, Mx_float, My_float = compute_capacities(
        Fc, Fr, Mcx, Mcy, Mrx, Mry)
    print(P_float)
    print(Mx_float)
    print(My_float)

    # Plot section for specific location of neutral axis
    plot_uls_section = 'Yes'
    if plot_uls_section == 'Yes':
        section_plot_uls.plot_ULS_section(
            x, y, xr, yr, x_sb, y_sb, Asb, sb_cog, Fc, Fr, Mcx, Mcy, Mrx, Mry, Mx_float, My_float, alpha_deg, na_y)

    Mx = [i/10**6 for i in Mx]
    My = [i/10**6 for i in My]
    P = [i/10**3 for i in P]

    #
    Ped = [1195]
    Mxed = [150]
    Myed = [200]

    point_cloud = np.transpose(np.array([P, Mx, My]))  # Assemble points to numpy array
    c_hull = ConvexHull(point_cloud)
    U = np.array([Ped[0], Mxed[0], Myed[0]])
    intersection = line_hull_intersection(U, c_hull)

    load_comb = np.array([Ped, Mxed, Myed])

    load_comb_dist = point_to_point_dist_3d([0, 0, 0], load_comb)
    capacity_dist = point_to_point_dist_3d([0, 0, 0], intersection)

    # print('load_comb_dist_single_test = ', load_comb_dist)
    # print('UR_single_test = ', load_comb_dist / capacity_dist)
    # print('cap_intersection_single_test = ', intersection)

    # Compute utilizations ratios
    ur = utilization_ratio(Ped, Mxed, Myed, P, Mx, My)
    # print('UR_test2_function =', ur)

    # print(in_point_cloud(point_cloud, load_comb))
    fig_surface = plt.figure()
    ax = Axes3D(fig_surface)
    scat = ax.scatter(Mx, My, P, linewidth=0.2, antialiased=True)
    ax.scatter(Mxed, Myed, Ped, color='green', s=35)
    ax.plot([0, intersection[1]], [0, intersection[2]], [0, intersection[0]], '-', color='purple')
    plt.show()




####################################################
# LOGGING STATEMENTS
####################################################
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
