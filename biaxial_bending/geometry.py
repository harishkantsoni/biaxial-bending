from math import sqrt, pi, cos, sin, tan, atan, atan2
import numpy as np


# ------------------------------------------
# 2D GEOMETRY
# ------------------------------------------

# Compute radius from 3 point on a circle
def radius_from_3_points():
    pass

# Compute intersection between line and polygon
def line_polygon_collisions(angle_deg, y_intersect, x_vertex, y_vertex):
    '''
    Return intersection points between a line and a polygon. If no intersections are present, return original polygon
    vertices.

    INPUT ARGUMENTS:
        angle           -   Angle of line with x-axis (in radians)
        y_intersect     -   Intersection between line and y-axis
        x_vertex        -   x-coordinates of polygon vertices
        y_vertex        -   y-coordinates of polygon vertices

    OUTPUT:
        xint            -   x-coordinates for intersections
        yint            -   y-coordinates for intersections

    '''
    # Convert input angle from [deg] to [rad]
    angle = angle_deg * pi / 180 

    # Evaluate the neutral axis linear equation for each vertex
    vertex_eval = [tan(angle)*x_vertex[i]-y_vertex[i]+y_intersect for i in range(len(x_vertex))]

    # TODO Code below can be shortened considerably by use of list comprehensions and the all() method
    # If corners are either all positive or all negative, the neutral axis is outside the cross section
    count_pos = 0
    count_neg = 0
    for vertex in vertex_eval:
        if vertex > 0:
            count_pos += 1
        else:
            count_neg += 1

    if count_neg == len(vertex_eval):
        # Neutral axis is located outside of cross section, return polygon vertices as output
        return x_vertex, y_vertex
        # TODO Logging! print('Neutral axis is outside of cross section - Entire section is in compression')
    elif count_pos == len(vertex_eval):
        # Neutral axis is located outside of cross section, return polygon vertices as output
        # TODO Logging! print('Neutral axis is outside of cross section - Entire section is in tension')
        return x_vertex, y_vertex
    else:
        # Neutral axis is inside the cross section (impliying two intersecting points with the section boundary)

        # Copy first entry to last entry to create a closed polygon
        vertex_eval.append(vertex_eval[0])
        xv = x_vertex + [x_vertex[0]]       # New local variable for vertices of closed polygon
        yv = y_vertex + [y_vertex[0]]       # New local variable for vertices of closed polygon

        xint = []           # Initiate list for holding two x-coordinates for intersections btw. neutral axis and section
        yint = []           # Initiate list for holding two y-coordinates for intersections btw. neutral axis and section
        for i in range(len(vertex_eval)-1):
            if np.sign(vertex_eval[i]) != np.sign(vertex_eval[i+1]):

                # NOTE DOES THIS ASSUME THAT VERTICES ARE ORDERED??????

                # Intersection detected
                # TODO Logging! print('Intersection detected between corner {} and {}'.format(i+1, i+2))    # Count starts from 0
                # Determine equation for line between corners (y = mx + k)
                x1 = xv[i]
                y1 = yv[i]
                x2 = xv[i+1]
                y2 = yv[i+1]

                if x1 == x2:
                    # Cross section line is vertical => slope is infinite. Line equation is just equal to x-coordinate
                    x = x1                                      # x-coordinate of intersection
                    y = tan(angle) * x + y_intersect            # y-coordinate of intersection
                    xint.append(x)
                    yint.append(y)
                else:
                    m = (y2 - y1) / (x2 - x1)                   # Slope of line btw. corner points
                    k = y2 - m * x2                             # Intersection with y-axis for line btw. corner points

                    x = (y_intersect - k) / (m - tan(angle))    # x-coordinate for intersection
                    y =  m * x + k                              # y-coordinate for intersection
                    xint.append(x)
                    yint.append(y)

    return xint, yint


# Calculate the area of a polygon by using the Shoelace Formula
def polygon_area(x, y, signed=False):
    ''' Return the area of a non-self-intersecting polygon given the coordinates of its vertices'''
    if x is not None and y is not None:
        # Copy coordinates of first point and append to list to create a closed polygon
        x = x + [x[0]]
        y = y + [y[0]]

        # Perform shoelace multiplication
        a1 = [x[i] * y[i+1] for i in range(len(x)-1)]
        a2 = [y[i] * x[i+1] for i in range(len(y)-1)]

        if signed == True:
            A = 1/2 * ( sum(a1) - sum(a2) )
        else:
            A = 1/2 * abs( sum(a1) - sum(a2) )

    else:
        A = 0

    return A

# Distance from point to line (optionally signed)
def point_to_line_dist(x, y, x0, y0, x1, y1, signed=True):
    '''
    Return the distance from a point (x, y) to a line passing through points (x0, y0) and (x1, y2).
    By default, the distance is 'signed', i.e. can be both positive and negative depending on point location compared
    to the line.
    The sign convention is chosen so a line placed on the x-axis, i.e. 0° with horizontal, has positive and negative
    distances to points located above and below it, respectively.
    '''
    # NOTE Function should provide the option of passing slope and y-intersection as input for representing the line
    if signed == True:
        return -( (y0 - y1) * x + (x1 - x0) * y + (x0 * y1 - x1 * y0) ) / sqrt((x1 - x0)**2 + (y1 - y0)**2)
    elif signed == False:
        return abs( (y0 - y1) * x + (x1 - x0) * y + (x0 * y1 - x1 * y0) ) / sqrt((x1 - x0)**2 + (y1 - y0)**2)


# Location of polygon centroid
def polygon_centroid(x, y, return_area=False):
    ''' Compute the centroid of a non-self-intersecting polygon given the coordinates of its vertices (corners)'''
    # Compute signed area of polygon
    A = polygon_area(x, y, signed=True)

    # TODO -----------
    # TODO In order for the formulas to work, the vertices must be en consecutive order along the polygon perimeter. _
    # TODO Make sure the function organizes the vertices if they are not already. The 'order_polygon_vertices' _
    # TODO function calls this function to order the vertices though. How to order them with unknown the centroid?
    # TODO -----------
    if A == 0:
        return np.nan
    else:
        # Copy coordinates of first point to last entry to create a closed polygon
        x = x + [x[0]]
        y = y + [y[0]]
        cx = []
        cy = []

        for i in range(len(x)-1):
            cx.append( ( x[i] + x[i+1] ) * ( x[i] * y[i+1] - x[i+1] * y[i] ) )
            cy.append( ( y[i] + y[i+1] ) * ( x[i] * y[i+1] - x[i+1] * y[i] ) )

        Cx = sum(cx) / (6*A)
        Cy = sum(cy) / (6*A)

        if return_area:
            return Cx, Cy, A
            
        return Cx, Cy


def order_polygon_vertices(x_vertices, y_vertices, x_section_vertices, y_section_vertices,
                           counterclockwise=True):
    '''
    Sort polygon vertices in consecutive circular order (clockwise or counterclockwise) measured from positive x-axis.
    
    Args:
    # TODO THIS SETUP IS NOT GOOD STYLE! THE PARAMS CALLED x_vertices AND y_vertices should default 
    #      to None and there should be a check to see if they were inputted.

        x_vertices         (list) : x-coordinate of vertices of polygon that needs to ber ordered
        y_vertices         (list) : y-coordinate of vertices of polygon that needs to ber ordered
        x_section_vertices (list) : x-coordinate of vertices of polygon whos centre of gravity 
                                    provides basis for ordering
        y_section_vertices (list) : x-coordinate of vertices of polygon whos centre of gravity 
                                    provides basis for ordering

    Returns:
        Lists of ordered coordinates
    '''
    x_t = x_vertices
    y_t = y_vertices

    # Compute centroid of entire section (containing section vertices)
    Cx, Cy = polygon_centroid(x_section_vertices, y_section_vertices)

    # Find angles of target vertices
    a0 = []     # Initialize lists for holding angles in original order
    for i in range(len(x_t)):
        a0.append( atan2( (y_t[i] - Cy),  (x_t[i] - Cx) ) * 180/pi )     # [deg]

    if counterclockwise == True:
        pos = [p for p in a0 if p >= 0]      # Positive angles
        neg = [n for n in a0 if n < 0]       # Negative angles

        # Make all angles positive and sort in counterclockwise order
        a = pos + [360+angle for angle in neg]

        # Get index of original entries after sorting
        idx = sorted( range(len(a0)), key= lambda j: a0[j], reverse=False )
        a = sorted(a)   # Sort angles
    else:
        # TODO Sort list of angles in clockwise order
        pass

    # Rearrange coordinate lists according to the specified order (clockwise or counterclockwise)
    x_t = [x_t[i] for i in idx]
    y_t = [y_t[i] for i in idx]

    return x_t, y_t


def get_section_compression_vertices(x, y, na_y, alpha_deg, delta_v):
    '''   Returns a list of the concrete section vertices that are in compression   '''

    # NOTE 'sb_eq_eval_at_each_vertex' could be computed in a simpler way perhaps
    def sb_eq_eval(angle, na_y, y_shift, x, y):
        '''Evaluation of equation for inner stress block at point (x, y)'''
        return tan(angle) * x - y + na_y - y_shift

    x_compr_vertices = []
    y_compr_vertices = []
    alpha = alpha_deg*pi/180
    for i in range(len(x)):
        # Evaluation of the stress block equation will determine the compression vertices
        sb_eq_eval_at_each_vertex = sb_eq_eval(alpha, na_y, delta_v, x[i], y[i])

        if alpha_deg <= 90 or alpha_deg > 270:
            if sb_eq_eval_at_each_vertex < 0:
                # logging.debug('Vertex at ({}, {}) is in compression'.format(x[i], y[i]))
                x_compr_vertices.append( x[i] )
                y_compr_vertices.append( y[i] )

        if alpha_deg > 90 and alpha_deg <= 270:
            if sb_eq_eval_at_each_vertex > 0:
                # logging.debug('Vertex at ({}, {}) is in compression'.format(x[i], y[i]))
                x_compr_vertices.append( x[i] )
                y_compr_vertices.append( y[i] )

    return x_compr_vertices, y_compr_vertices


# ------------------------------------------
# 3D GEOMETRY
# ------------------------------------------

def point_to_point_dist_3d(P1, P2):
    return sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2 + (P2[2]-P1[2])**2)


def line_hull_intersection(U, c_hull):
    ''' 
    Return intersection between the line defined by vector U with convex hull

    Args:
        - U (numpy array)   : Vector defining line, format 'U=np.array([X, Y, Z])' 
        - c_hull (object)   : Scipy object returned from a convex hull analysis 
    '''
    eq = c_hull.equations.T
    V, b = eq[:-1], eq[-1]
    V = np.transpose(V)
    alpha = -b / np.dot(V, U)
    return np.min(alpha[alpha > 0]) * U


if __name__ == '__main__':
    pass
