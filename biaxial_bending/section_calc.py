# Builtin libraries
from math import pi, cos, sin, tan, atan, atan2, sqrt, ceil, floor

# Third party libraries
import numpy as np

# Project specific modules
import geometry

'''
This module contains functions that can be used for reinforced concrete section analysis in both SLS and ULS.

'''


def compute_plastic_centroid(x, y, xr, yr, As, fck, fyk):

    Ac = geometry.polygon_area(x, y)
    eta = 0.85
    F = sum([As[i]*fyk for i in As]) + eta*(Ac - sum(As))*fck

    # TODO Find correct and general arm for concrete force (polygon section)
    F_times_dx = sum([As[i]*fyk*xr[i]
                      for i in range(len(xr))]) + eta*(Ac - sum(As))*fck*500/2
    F_times_dy = sum([As[i]*fyk*yr[i]
                      for i in range(len(yr))]) + eta*(Ac - sum(As))*fck*375/2

    xpl = F_times_dx/F
    ypl = F_times_dy/F

    return xpl, ypl


def compute_dist_from_na_to_vertices(x, y, xr, yr, alpha_deg, na_y):

    alpha = alpha_deg * pi / 180    # [rad]

    # FIXME dv fails for pure compression
    # Find 'signed' distances from neutral axis to each vertex
    dv = []         # Initialize list for holding distances from each vertex to neutral axis

    # Define two known points on line representing neutral axis
    # NOTE Could be done more elegant if the function allowed tan(alpha) and na_y as input
    na_x0 = 0
    na_y0 = tan(alpha) * na_x0 + na_y
    na_x1 = 1
    na_y1 = tan(alpha) * na_x1 + na_y

    # Compute distances from neutral axis to each vertex (neg. value => vertex in compr. / pos. value => vertex in tension)
    # FIXME FIX DISTANCES FOR TENSION/COMPRESSION!!!
    for i in range(len(x)):
        dv.append(geometry.point_to_line_dist(
            x[i], y[i], na_x0, na_y0, na_x1, na_y1))

    dr = []      # Distances btw. each rebar and neutral axis
    for i in range(len(xr)):
        dr.append(geometry.point_to_line_dist(
            xr[i], yr[i], na_x0, na_y0, na_x1, na_y1))

    # Reverse sign of the 'signed' distances if slope of neutral axis becomes negative
    if alpha_deg > 90 and alpha_deg <= 270:
        dv = list(np.negative(dv))
        dr = list(np.negative(dr))

    # Change potential distances of '-0.0' to '0.0' to avoid getting the wrong cross section state later
    dv = [0.0 if x == -0.0 else x for x in dv]

    return dv, dr


def compute_stress_block_geomemtry(x, y, dv, dr, alpha_deg, na_y, beta_1=0.85):
    '''
    Returns stress block geometry

    INPUT
        x           -   List of x-coordinates of concrete section vertices
        y           -   List of y-coordinates of concrete section vertices
        dv          -   List of distances from neutral axis to each section vertex
        dr          -   List of distances from neutral axis to each rebar
        alpha_deg   -
        na_y        -

    OUTPUT
        x_sb        -   List of x-coordinates of stress block vertices
        y_sb        -   List of y-coordinates of stress block vertices
        Asb         -   Area of stress block
        sb_cog      -   Cenntroid of stress block represented as tuple, i.e. in the format (x, y)
    '''

    # PURE TENSION CASE
    # NOTE Test is this is true! Does not account for  gap btw. sb and tension zone
    if all(d >= 0 for d in dv):
        cross_section_state = 'PURE TENSION'

        # Distance from neutral axis to extreme tension bar (all distances will be positve)
        c = max([d for d in dr if d > 0])

        # Set vertices of stress block
        x_sb = None
        y_sb = None

        # Set stress block area
        Asb = 0

        sb_cog = None

    # PURE COMPRESSION CASE
    elif all(d <= 0 for d in dv):   # NOTE Test if this is true!
        cross_section_state = 'PURE COMPRESSION'

        # Distance from neutral axis to extreme compression fiber (all distances will be negative)
        c = min(dv)

        # Set vertices of stress block (entire section)
        x_sb = x
        y_sb = y

        Asb = geometry.polygon_area(x, y)
        sb_cog = geometry.polygon_centroid(x, y)

    # MIXED TENSION/COMPRESSION CASE
    else:
        cross_section_state = 'MIXED TENSION/COMPRESSION'

        # Distance from neutral axis to extreme compression fiber (pos. in tension / negative in compression)
        # FIXME This might not be correct in all cases (if compression zone is very small, tension will dominate)
        c = min(dv)

        # FIXME Fix naming below
        # NOTE beta_1=0.85 from ACI should be replaced by lambda = 0.8 from Eurocode for concrete strengths < C50 (change also default function input)
        # Distance from inner stress block edge to extreme compression fiber
        a = beta_1 * c
        # Perpendicular distance between neutral axis and stress block
        delta_p = c - a
        alpha = alpha_deg*pi/180
        # Vert. dist. in y-coordinate from neutral axis to inner edge of stress block
        delta_v = delta_p / cos(alpha)

        # Intersection between stress block inner edge and y-axis
        sb_y_intersect = na_y - delta_v

        # Intersections between stress block and section
        sb_xint, sb_yint = geometry.line_polygon_collisions(
            alpha, sb_y_intersect, x, y)

        x_compr_vertices, y_compr_vertices = geometry.get_section_compression_vertices(
            x, y, na_y, alpha, delta_v)

        # Collect all stress block vertices
        x_sb = sb_xint + x_compr_vertices
        y_sb = sb_yint + y_compr_vertices

        # Order stress block vertices with respect to centroid for the entire section
        # NOTE Might fail for non-convex polygons, e.g. a T-beam
        x_sb, y_sb = geometry.order_polygon_vertices(
            x_sb, y_sb, x, y, counterclockwise=True)

        # Compute area of the stress block by shoelace algorithm
        Asb = geometry.polygon_area(x_sb, y_sb)

        # Compute location of centroid for stress block polygon
        sb_cog = geometry.polygon_centroid(x_sb, y_sb)

    return x_sb, y_sb, Asb, sb_cog, c


def compute_rebar_strain(dist_to_na, c, eps_cu):
    '''    Returns strain in each rebar as a list    '''
    return [ri / abs(c) * eps_cu for ri in dist_to_na]


def compute_rebar_stress(eps_r, Es, fyk):
    '''    Returns stress in each rebar as a list    '''
    # NOTE Could be expanded to handle both 'ULS' and 'SLS'
    sigma_r = []
    for i in range(len(eps_r)):
        # Linear elastic stress in i'th bar
        si = eps_r[i] * Es
        if abs(si) <= fyk:
            # Use computed stress if it does not exceed yielding stress
            sigma_r.append(si)
        else:
            # If computed stress exceeds yield, use yielding stress instead
            sigma_r.append(np.sign(si)*fyk)

    return sigma_r


def get_rebars_in_stress_block(xr, yr, x_sb, y_sb):
    '''    Returns a list with entry 'True' for rebars located inside the stress block, 'False' otherwise    '''
    # Arrange rebar coordinates
    rebar_coords = [[xr[i], yr[i]] for i in range(len(xr))]

    # Arrange stress block coordinates
    Asb = geometry.polygon_area(x_sb, y_sb)
    if Asb != 0:
        sb_poly = [[x_sb[i], y_sb[i]] for i in range(len(x_sb))]

        # Check if rebars are inside the stress block
        path = mpltPath.Path(sb_poly)
        # Returns 'True' if rebar is inside stress block
        rebars_inside = path.contains_points(rebar_coords)
    else:
        # All rebars are in tension (all entries are 'False')
        rebars_inside = [False] * len(xr)

    return rebars_inside
    # logging.debug('bar {} is inside stress block'.format(i+1))  # TODO Create logging statement


def compute_rebar_forces(xr, yr, As, sigma_r, rebars_inside):
    ''' Return rebar forces as list'''
    Fr = []    # Forces in each rebar

    for i in range(len(xr)):
        if rebars_inside[i] == True:
            # Rebar is inside stress block, correct for disp. of concrete
            Fi = (sigma_r[i] + 0.85 * fcd) * As
        else:
            Fi = sigma_r[i] * As
        Fr.append(Fi)

    return Fr


def compute_concrete_force(fck, gamma_c, Asb):
    ''' Return compression force in the concrete. '''
    Fc = -0.85 * fck/gamma_c * Asb
    return Fc


def compute_moment_vector_angle(Mx, My):
    '''    Returns the angle (in degrees) of the moment vector with respect to the x-axis    '''
    if Mx == 0:
        if My == 0:
            phi = None
        else:
            phi = 90
    else:
        phi = atan(My/Mx)*180/pi

    return phi


def compute_C_T_forces(Fc, Fr):
    '''    Returns Compression (C) and Tension (T) forces of the section    '''
    Fr_compr = [p for p in Fr if p <= 0]
    Fr_tension = [p for p in Fr if p > 0]
    C = sum(Fr_compr) + Fc
    T = sum(Fr_tension)

    return C, T


def compute_C_T_moments(C, T, Mcx, Mcy, Mry, Mrx, Fr):
    '''
    Returns total moments generated in the section by Compression (C) and Tension (T) resisting forces.

    The calculation assumes a left-handed sign convention.
    '''
    # TODO Change loop below to list comprehensions
    My_compr = []
    Mx_compr = []
    My_tension = []
    Mx_tension = []
    for i in range(len(Fr)):
        if Fr[i] < 0:
            My_compr.append(Mry[i])
            Mx_compr.append(Mrx[i])
        if Fr[i] > 0:
            My_tension.append(Mry[i])
            Mx_tension.append(Mrx[i])

    # Total moment for compression resisting forces (adapted for LH sign convention)
    if alpha_deg >= 90 and alpha_deg <= 270:
        My_C = sum(My_compr) + Mcy
        Mx_C = sum(Mx_compr) + Mcx
    else:
        My_C = -(sum(My_compr) + Mcy)
        Mx_C = -(sum(Mx_compr) + Mcx)

    # Total moment for tension resisting forces (adapted for LH sign convention)
    if alpha_deg >= 90 and alpha_deg <= 270:
        My_T = sum(My_tension)
        Mx_T = sum(Mx_tension)
    else:
        My_T = -sum(My_tension)
        Mx_T = -sum(Mx_tension)

    return Mx_C, My_C, Mx_T, My_T


def compute_C_T_forces_eccentricity(C, T, My_C, Mx_C, Mx_T, My_T):
    '''    Return eccentricity of Compression (C) and Tension (T) forces.    '''
    # Eccentricities of tension and compression forces
    if C == 0:
        ex_C = np.nan
        ey_C = np.nan
    else:
        ex_C = My_C/C
        ey_C = Mx_C/C

    if T == 0:
        ex_T = np.nan
        ey_T = np.nan
    else:
        ex_T = My_T/T
        ey_T = Mx_T/T

    return ex_C, ey_C, ex_T, ey_T
