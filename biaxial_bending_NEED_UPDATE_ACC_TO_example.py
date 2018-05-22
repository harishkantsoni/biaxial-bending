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

# SIGN CONVENTION
# Coordinate system :   left handed
# Tension           :   Positive
# Compression       :   Negative


# Create log file and set default logging statement
FORMAT = '%(name)-15s %(message)s'
filename = 'test.log'
logging.basicConfig(filename=filename, level=logging.DEBUG, filemode='w', format=FORMAT)

# DESCRIPTION
# - The term 'stress block' is used for the concrete area that, in the calculations, is assumed to be under compression.
#   This area depends on the adopted stress-strain curve of the concrete, where the most used one is the Whitney Stress
#   Block.
#   In reality, everything on one side of the neutral axis is compression.

# -------------- FIND INTERSECTIONS BTW. NEUTRAL AXIS AND SECTION BOUNDARIES -----------
#
#       Equation representing neutral axis:                     0 = tan(alpha) * x - y + na_y
#
#       Equation representing inner edge of strees block        0 = tan(alpha) * x - y + na_y - delta_v
#
# where
#   - 'delta_v' is the vertical distance between the neutral axis and the inner stress block edge
#   - 'na_y' is the y-ccordinate of the intersection btw. the y-axis and the neutral axis.
#

np.set_printoptions(precision=2)

# TODO No asumption should be made about location of Origo. Right now it is at middle of section, which is only good
# for rectangular shapes.
# FIXME If entire cross section is in compression, the plot should have all the section be grey

b = 16      # [in]
h = 16      # [in]

Ac = b*h    # Area of concrete section TODO Should be computed for at polygon instead

eps_cu = 0.003      # Compressive crushing strain of concrete (strain when cross section capacity is reached)

fck =  4    # [ksi]
gamma_c = 1.0
fcd = fck/gamma_c

Es = 29000  # [ksi]
fy = 60     # [ksi]

# Concrete geometry defined by polygon vertices
x = [-b/2, b/2, b/2, -b/2]
y = [h/2, h/2, -h/2, -h/2]

# Rebar locations
xr = [-5.6, 0,   5.6, 5.6,  5.6, 0,    -5.6, -5.6]
yr = [ 5.6, 5.6, 5.6,  0,  -5.6, -5.6,  -5.6, 0]

Ø = 1

# As = pi * (Ø / 2)**2   # [in^2]    FIXME This is only the area of a single bar
As = 0.79

beta_1 = 0.85      # Factor for compression zone height of Whitney stress block

nbars = len(xr)

# alpha_deg = 30                  # [deg]
# na_y = -2       # [in] Distance from top of section to intersection btw. neutral axis and y-axis
# # NOTE na_y Should be infinite if alpha is 90 or 270

# TODO Find a better way to represent increments for na_y, right now 0 is being computed twice
# TODO Stop varying na_y if pure tension or compression is found, i.e. if the moment capacities both become 0
n = 2
na_y_list = [-b*4/2*i/n for i in range(n)] + [b*4/2*i/n for i in range(n)]
# na_y_list = [-5]
alpha_list = [alpha for alpha in range(0, 360, 20)]
# alpha_list = [0]
print('Number of computations: {}'.format(len(na_y_list)*len(alpha_list)))

P_list = []
Mx_list = []
My_list = []
na_y_computed =[]
angles_computed = []

for na_y in na_y_list:
    for alpha_deg in alpha_list:
        angles_computed.append(alpha_deg)
        na_y_computed.append(na_y)
        alpha = alpha_deg * pi / 180    # [rad]

        # Find collision points between neutral axis and concrete section
        na_xint, na_yint = geometry.line_polygon_collisions(alpha, na_y, x, y)

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
            dv.append( geometry.point_to_line_dist(x[i], y[i], na_x0, na_y0, na_x1, na_y1) )

        rp = []      # Perpendicular distances btw. each rebar and neutral axis
        for i in range(nbars):
            rp.append( geometry.point_to_line_dist(xr[i], yr[i], na_x0, na_y0, na_x1, na_y1) )

        # Reverse sign of the 'signed' distances if slope of neutral axis becomes negative
        if alpha_deg > 90 and alpha_deg <= 270:
            dv = list(np.negative(dv))
            rp = list(np.negative(rp))

        # ---------- STRESS BLOCK GEOMETRY

        # # Change potential distances of '-0.0' to '0.0'. This is to avoid getting the wrong cross section state below.
        # dv = [0.0 if x==-0.0 else x for x in dv]

        # PURE TENSION CASE
        if all(d >= 0 for d in dv):
            # Entire cross section is in tension NOTE Test is this is true! This does not account for the gap btw. sb and tension zone
            cross_section_state = 'PURE TENSION'

            # Distance from neutral axis to extreme tension bar (all distances will be positve)
            c = max([r for r in rp if r > 0])

            # Set vertices of stress block
            x_sb = None
            y_sb = None

            Asb = 0

        # PURE COMPRESSION CASE
        elif all(d <= 0 for d in dv):
            # Entire cross section is in compression    NOTE Test if this is true!
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
            # Cross section has a mix of compression and tension
            cross_section_state = 'MIXED TENSION/COMPRESSION'

            # Distance from neutral axis to extreme compression fiber (pos. in tension / negative in compression)
            # FIXME This might not be correct in all cases (if compression zone is very small, tension will dominate)
            c = min(dv)

            a = beta_1 * c                      # Distance from inner stress block edge to extreme compression fiber
            delta_p = c - a                     # Perpendicular distance between neutral axis and stress block
            delta_v = delta_p / cos(alpha)     # Vert. dist. in y-coordinate from neutral axis to inner edge of stress block

            sb_y_intersect = na_y - delta_v     # Intersection between stress block inner edge and y-axis

            # Intersections between stress block and section
            sb_xint, sb_yint = geometry.line_polygon_collisions(alpha, sb_y_intersect, x, y)

            def sb_eq_eval(angle, na_y, y_shift, x, y):
                '''Evaluation of equation for inner stress block at point (x, y)'''
                return tan(angle) * x - y + na_y - y_shift

            # Determine vertices in compression
            x_compr_vertices = []
            y_compr_vertices = []
            for i in range(len(x)):
                # Evaluation of the stress block equation will determine the compression vertices
                sb_eq_eval_at_each_vertex = sb_eq_eval(alpha, na_y, delta_v, x[i], y[i])

                if alpha_deg < 90 or alpha_deg > 270:
                    if sb_eq_eval_at_each_vertex < 0:
                        logging.debug('Vertex at ({}, {}) is in compression'.format(x[i], y[i]))
                        x_compr_vertices.append( x[i] )
                        y_compr_vertices.append( y[i] )

                if alpha_deg >= 90 and alpha_deg <= 270:
                    if sb_eq_eval_at_each_vertex > 0:
                        x_compr_vertices.append( x[i] )
                        y_compr_vertices.append( y[i] )
                        logging.debug('Vertex at ({}, {}) is in compression'.format(x[i], y[i]))

            # Get stress block vertices by collecting compression corners and intersections btw. stress block inner edge and section
            x_sb = sb_xint + x_compr_vertices
            y_sb = sb_yint + y_compr_vertices

            # Order stress block vertices with respect to centroid for the entire section
            # NOTE Might fail for non-convex polygons, e.g. a T-beam
            x_sb, y_sb = geometry.order_polygon_vertices(x_sb, y_sb, x, y, counterclockwise=True)

            # Compute area of the stress block by shoelace algorithm
            Asb = geometry.polygon_area(x_sb, y_sb)     # Area of stress block

            # FIXME sb_cog er forkert for na_y=-2 og alpha_deg=91
            # Compute location of centroid for stress block polygon
            sb_cog = geometry.polygon_centroid(x_sb, y_sb)

        # The sign convention is chosen as positve for tension and negative for compression. This means that the distances
        # computed by geometric considerations need to swap signs.

        # TODO There might be smarter ways to fill lists for stress and strain
        eps_r = []  # Strain in each rebar
        for i in range(nbars):
            eps_r.append(rp[i] / abs(c) * eps_cu)

        # Compute rebar stress
        sigma_r = []
        for i in range(nbars):
            si = eps_r[i] * Es
            if abs(si) <= fy:
                sigma_r.append(si)   # Use computed stress if it does not exceed yielding stress
            else:
                sigma_r.append(np.sign(si)*fy)  # If computed stress exceeds yield, use yielding stress instead


        # NOTE Arragement of coordinates and check for rebars inside stress block could be done better/in fewer lines
        # Arrange rebar coordinates
        rebar_coords = []
        for i in range(len(xr)):
            rebar_coords.append([xr[i], yr[i]])

        # Arrange stress block coordinates
        if Asb != 0:
            sb_poly = []
            for i in range(len(x_sb)):
                sb_poly.append([x_sb[i], y_sb[i]])

            # Check if rebars are inside the stress block
            path = mpltPath.Path(sb_poly)
            rebar_inside = path.contains_points(rebar_coords)   # Returns 'true' if rebar is inside stress block
        else:
            # All rebars are in tension (all entries are 'False')
            rebar_inside = [False] * len(xr)

        # Compute rebar forces
        Fr = []    # Forces in each rebar
        for i in range(len(xr)):
            if rebar_inside[i] == True:  # If rebar is inside stress block, correct for disp. of concrete
                logging.debug('bar {} is inside stress block'.format(i+1))
                Fi = (sigma_r[i] + 0.85 * fcd) * As
            else:
                Fi = sigma_r[i] * As
            Fr.append(Fi)

        Fc = -0.85 * fcd * Asb   # Compression force in concrete

        # Compute moment resistances
        Mrx = []    # Moment contribution from rebars for bending about x-axis
        Mry = []    # Moment contribution from rebars for bending about y-axis
        for i in range(nbars):
            # FIXME Lever arms should be taken wrt. the centroid of the transformed section, i.e. including reinforcement
            # Correct for moment sign convention (see description in the beginning for coordinate system)
            if alpha_deg >= 90 and alpha_deg <= 270:
                Mrx.append(-Fr[i] * yr[i])
                Mry.append(-Fr[i] * xr[i])
            else:
                Mrx.append(-Fr[i] * yr[i])
                Mry.append(-Fr[i] * xr[i])

        if Asb == 0:
            Mcx = 0
            Mcy = 0
        else:
            # FIXME Moment lever arm should be distance between stress block centroid and centroid of transformed section
            if alpha_deg >= 90 and alpha_deg <= 270:
                Mcx = -Fc * sb_cog[1]    # Moment contribution from concrete in x-direction
                Mcy = -Fc * sb_cog[0]    # Moment contribution from concrete in y-direction
            else:
                Mcx = -Fc * sb_cog[1]    # Moment contribution from concrete in x-direction
                Mcy = -Fc * sb_cog[0]    # Moment contribution from concrete in y-direction

        # Total capacities
        P = sum(Fr) + Fc
        Mx = sum(Mrx) + Mcx
        My = sum(Mry) + Mcy

        # Angle of resulting moment vector with x-axis
        if Mx == 0:
            if My == 0:
                phi = None
            else:
                phi = 90
        else:
            phi = atan(My/Mx)*180/pi

        Fr_compr = [p for p in Fr if p <= 0]
        Fr_tension = [p for p in Fr if p > 0]

        C = sum(Fr_compr) + Fc
        T = sum(Fr_tension)

        My_compr = []
        Mx_compr = []
        My_tension = []
        Mx_tension = []
        for i in range(nbars):
            if Fr[i] < 0:
                My_compr.append(Mry[i])
                Mx_compr.append(Mrx[i])
            if Fr[i] > 0:
                My_tension.append(Mry[i])
                Mx_tension.append(Mrx[i])

        # Total moment for compression resisting forces (adapted for sign convention)
        if alpha_deg >= 90 and alpha_deg <= 270:
            My_C = sum(My_compr) + Mcy
            Mx_C = sum(Mx_compr) + Mcx
        else:
            My_C = -(sum(My_compr) + Mcy)
            Mx_C = -(sum(Mx_compr) + Mcx)

        # Total moment for tension resisting forces (adapted for sign convention)
        if alpha_deg >= 90 and alpha_deg <= 270:
            My_T = sum(My_tension)
            Mx_T = sum(Mx_tension)
        else:
            My_T = -sum(My_tension)
            Mx_T = -sum(Mx_tension)

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

        P_list.append(P)
        Mx_list.append(Mx)
        My_list.append(My)

df = pd.DataFrame({'na_y': na_y_computed, 'alpha': angles_computed, 'P': P_list, 'Mx': Mx_list, 'My': My_list})
df = df[['P', 'Mx', 'My', 'alpha', 'na_y']]     # Set order of columns
df = df.sort_values(['na_y', 'alpha'], ascending=[True, True])
df = df.reset_index(drop=True)
df = df.round(decimals=2)
df.to_csv('df_results.txt')

print(df.to_string())

# TRI-SURFACE PLOT
fig_surface = plt.figure()
ax = Axes3D(fig_surface)
surf = ax.plot_trisurf(df.P, df.Mx, df.My, linewidth=0.2, antialiased=True)
# ax.set_xlabel(Mx)
# plt.ylabel(My)
# plt.zlabel(P)

# WIREFRAME PLOT
fig_surface = plt.figure()
ax2 = Axes3D(fig_surface)
wire = ax2.plot_wireframe(df.P, df.Mx, df.My, linewidth=0.2, antialiased=True)

# SCATTER PLOT
fig_surface = plt.figure()
ax3 = Axes3D(fig_surface)
scat = ax3.scatter(df.Mx, df.My, df.P, linewidth=0.2, antialiased=True)
ax3.set_xlabel('Mx')
ax3.set_ylabel('My')
ax3.set_zlabel('P')

plt.show()



# IDEA Make a dataframe from all relevant results from all iterations. These can be used for plotting any given result _
# afterwards
# IDEA Make an other script with the example from MacGregor's book, so that it is easily accessable for reference
# TODO Find a way to determine sign of the moment capacities


#####################################################
# PLOT RESULTS
#####################################################
# fig, ax = plt.subplots()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.style.use('seaborn-white')
#
# # Full concrete section
# # ax1.add_patch(patches.Rectangle((-b/2, -h/2), b, h, facecolor='white', edgecolor='k', linewidth=1.5))
#
# plt.plot([-b/2, -b/2, b/2, b/2, -b/2], [-h/2, h/2, h/2, -h/2, -h/2], '-', color='k', linewidth=1)    # Concrete section
#
# # Coordinate axes
# plt.plot([-1.2*b/2, 1.2*b/2], [0, 0], 'k', linewidth=0.3)
# plt.plot([0, 0], [-1.2*h/2, 1.2*h/2], 'k', linewidth=0.3)
# plt.annotate('$x$', (b/2+b/8, 0), verticalalignment='center')
# plt.annotate('$y$', (0, h/2+h/8), horizontalalignment='center')
#
# plt.plot(na_xint, na_yint, color='k', linewidth=1)      # Plot neutral axis
#
# plt.plot([ex_C, ex_T], [ey_C, ey_T], '.', color='b')    # Resulting compression and tension force of the section
# plt.annotate('$C$', (ex_C, ey_C), color='b', horizontalalignment='left',
#                                     verticalalignment='center', fontsize=12)
# plt.annotate('$T$', (ex_T, ey_T), color='b', horizontalalignment='left',
#                                     verticalalignment='center', fontsize=12)
#
# # Plot stress block
# if Asb != 0:
#     # Create list of stress block coords. in the format [[x0, y0], [x1, y2], ..., [xn, yn]] for plotting as a polygon patch
#     sb_coords = []
#     if Asb == Ac:    # Pure compression
#         sb_coords = [x, y]
#     else:
#         for i in range(len(x_sb)):
#             sb_coords.append( [x_sb[i], y_sb[i]] )
#         ax.add_patch(patches.Polygon((sb_coords), facecolor='silver', edgecolor='k', linewidth=1))
# print(x_sb)
# print(y_sb)
# print(sb_coords)
# print('asdasdjakj')
# # Plot centroid of stress block
# if Asb != 0:
#     plt.plot(sb_cog[0], sb_cog[1], 'x', color='grey', markersize='4')
#
# # TODO Radius of rebars on the plot should match the actual radius of the bars
# # Plot rebars
# for i in range(len(xr)):
#     ax.add_patch(patches.Circle((xr[i], yr[i]), radius=0.4, hatch='/////', facecolor='silver', edgecolor='k', linewidth=1))
#
# margin = b/4
# plt.axis((-(b/2+margin), b/2+margin, -(h/2+margin), h/2+margin))    # Set axis limits
# plt.show()
#
# print_stuff = 'Yes'
# if print_stuff == 'Yes':
#     print('x-intersections btw. sb and section: ' + str(sb_xint))
#     print('y-intersections btw. sb and section: ' + str(sb_yint))
#     print('x_compr_vertices: ' + str(x_compr_vertices))
#     print('y_compr_vertices: ' + str(y_compr_vertices))
#     print('x_sb: {}'.format(x_sb))
#     print('y_sb: {}'.format(y_sb))
#     print('Lever arm')
#     print(np.around(rp, decimals=2))
#     print('Strain - Rebars')
#     print(np.around(eps_r, decimals=5))
#     print('Stress - Rebars')
#     print(np.around(sigma_r, decimals=2))
#     print('Forces - Rebars')
#     print(np.around(Fr, decimals=2))
#     print('Forces - Concrete')
#     print(np.around(Fc, decimals=2))
#     print('sb_cog           ' + str(np.around(sb_cog, decimals=2)))
#     print('Mcx              ' + str(np.around(Mcx, decimals=2)))
#     print('Mcy              ' + str(np.around(Mcy, decimals=2)))
#     print('(P, Mx, My)      ' + str((np.around(P, decimals=2), np.around(Mx, decimals=2), np.around(My, decimals=2))))
#     print('phi              ' + str(np.around(phi, decimals=2)))
#     print('C:               ' + str(C))
#     print('T:               ' + str(T))
#     print('My_compr:        ' + str(My_compr))
#     print('Mx_compr:        ' + str(Mx_compr))
#     print('My_C:            ' + str(My_C))
#     print('Mx_C:            ' + str(Mx_C))
#     print('ey_C:            ' + str(ey_C))
#     print('ex_C:            ' + str(ex_C))
#     print('My_T:            ' + str(My_T))
#     print('Mx_T:            ' + str(Mx_T))
#     print('ey_T:            ' + str(ey_T))
#     print('ex_T:            ' + str(ex_T))
