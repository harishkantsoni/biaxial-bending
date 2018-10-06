import matplotlib.pyplot as pyplot

import geometry
import section_calc_ULS as sc



def plot_capacity_surface(X, Y, Z, plot_type='scatter', labels=['Mx', 'My', 'P']):
    '''    Plots capacity surface as 3D graph    '''
    fig_surface = plt.figure()
    ax = Axes3D(fig_surface)

    if plot_type == 'trisurf':
        surf = ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)

    if plot_type == 'wireframe':
        wire = ax.plot_wireframe(X, Y, Z, linewidth=0.2, antialiased=True)

    else:
        scat = ax.scatter(X, Y, Z, linewidth=0.2, antialiased=True)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.show()


def plot_ULS_section(x, y, xr, yr, na_y, alpha_deg):
    '''    Returns a plot of ULS section state for given neutral axis location    '''
    dv, dr = sc.compute_dist_from_na_to_vertices(x, y, xr, yr, alpha_deg, na_y)
    x_sb, y_sb, Asb, sb_cog, c = sc.compute_stress_block_geomemtry(dv, dr, alpha_deg, na_y)
    eps_r = sc.compute_rebar_strain(dr, c, eps_cu)
    sigma_r = sc.compute_rebar_stress(eps_r, Es, fyk)
    rebars_inside = sc.get_rebars_in_stress_block(xr, yr, x_sb, y_sb)
    P, Mx, My = sc.compute_capacities(xr, yr, Fr, Fc, sb_cog)
    phi = sc.compute_moment_vector_angle(Mx, My)
    C, T = sc.compute_C_T_forces(Fc, Fr)         
    Mx_C, My_C, Mx_T, My_T = sc.compute_C_T_moments(C, T, Mcx, Mcy, Mry, Mrx, Fr)
    ex_C, ey_C, ex_T, ey_T = sc.compute_C_T_forces_eccentricity(C, T, My_C, Mx_C, Mx_T, My_T)

    # Find collision points between neutral axis and concrete section # NOTE Only for plotting puposes, not used in calc
    na_xint, na_yint = geometry.line_polygon_collisions(alpha, na_y, x, y)

    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.style.use('seaborn-white')

    # Full concrete section
    x_plot = x
    x_plot.append(x_plot[0])
    y_plot = y
    y_plot.append(y_plot[0])
    plt.plot(x_plot, y_plot, '-', color='k', linewidth=1)    # Concrete section

    # Coordinate axes
    plt.plot([-1.2*b/2, 1.2*b/2], [0, 0], 'k', linewidth=0.3)       # TODO Should be more general, maybe pass thoguh plastic centroid?
    plt.plot([0, 0], [-1.2*h/2, 1.2*h/2], 'k', linewidth=0.3)
    plt.annotate('$x$', (b/2+b/8, 0), verticalalignment='center')
    plt.annotate('$y$', (0, h/2+h/8), horizontalalignment='center')

    plt.plot(na_xint, na_yint, color='k', linewidth=1)      # Plot neutral axis

    plt.plot([ex_C, ex_T], [ey_C, ey_T], '.', color='b')    # Resulting compression and tension force of the section
    plt.annotate('$C$', (ex_C, ey_C), color='b', horizontalalignment='left',
                                        verticalalignment='center', fontsize=12)
    plt.annotate('$T$', (ex_T, ey_T), color='b', horizontalalignment='left',
                                        verticalalignment='center', fontsize=12)

    # Plot stress block
    if Asb != 0:
        # Create list of stress block coords. in the format [[x0, y0], [x1, y2], ..., [xn, yn]] for plotting as a polygon patch
        # TODO List comprehension!
        sb_coords = []
        for i in range(len(x_sb)):
            sb_coords.append( [x_sb[i], y_sb[i]] )
        ax.add_patch(patches.Polygon((sb_coords), facecolor='silver', edgecolor='k', linewidth=1))

    # Plot centroid of stress block
    if Asb != 0:
        plt.plot(sb_cog[0], sb_cog[1], 'x', color='grey', markersize='4')

    # TODO Radius of rebars on the plot should match the actual radius of the bars
    # Plot rebars
    for i in range(len(xr)):
        ax.add_patch(patches.Circle((xr[i], yr[i]), radius=0.4, hatch='/////', facecolor='silver', edgecolor='k', linewidth=1))

    margin = b/4
    plt.axis((-(b/2+margin), b/2+margin, -(h/2+margin), h/2+margin))    # Set axis limits
    plt.show()


if __name__ == '__main__':

     # Define concrete geometry by polygon vertices
    x = [-8, 8, 8, -8]
    y = [8, 8, -8, -8]

    # Define rebar locations and sizes
    xr = [-5.6, 0,   5.6,  5.6,  5.6,  0,   -5.6, -5.6]
    yr = [ 5.6, 5.6, 5.6,  0,   -5.6, -5.6, -5.6,  0]

    Ø = 1
    As = 0.79
    beta_1 = 0.85      # Factor for compression zone height of Whitney stress block

    # FIXME ZeroDivisionError in 'eps_r.append(dr[i] / c * eps_cu)' for (alpha, na_y)=(45, -16) or (0, -8) ___
    # FIXME ___ Happens just as the section goes from almost pure compression to pure compression. See plot!

    # FIXME Compression zone is not computed correctly if na is below section, FIX!!! Same problem as above comment I think!
    alpha_deg = 15               # [deg]
    na_y = -2       # [in] Distance from top of section to intersection btw. neutral axis and y-axis
    # NOTE na_y Should be infinite if alpha is 90 or 270

    na_y = 0
    alpha_deg = 90

    # TODO This should be done in the form of a proper test instead
    plot_ULS_section(x, y, xr, yr, na_y, alpha_deg)


