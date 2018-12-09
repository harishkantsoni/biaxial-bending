import geometry
from math import tan, pi
import numpy as np
import matplotlib.pyplot as plt

b = 16
h = 16

alpha_deg = 0                  # [deg]
na_y = -9       # [in] Distance from top of section to intersection btw. neutral axis and y-axis
# NOTE na_y Should be infinite if alpha is 90 or 270

alpha = alpha_deg * pi / 180    # [rad]

# Concrete geometry defined by polygon vertices
x = [-b/2, b/2, b/2, -b/2]
y = [h/2, h/2, -h/2, -h/2]

# Rebar locations
xr = [-5.6, 0,   5.6, 5.6,  5.6, 0,    -5.6, -5.6]
yr = [ 5.6, 5.6, 5.6,  0,  -5.6, -5.6,  -5.6, 0]

na_x0 = 0
na_y0 = tan(alpha) * na_x0 + na_y
na_x1 = 1
na_y1 = tan(alpha) * na_x1 + na_y
dv = []
for i in range(len(x)):
    dv.append( geometry.point_to_line_dist(x[i], y[i], na_x0, na_y0, na_x1, na_y1) )

print(dv)

# Reverse sign of the 'signed' distances if slope of neutral axis becomes negative.
if alpha_deg > 90 and alpha_deg <= 270:
    dv = list(np.negative(dv))

print(dv)

plt.figure()
plt.plot([-b/2, -b/2, b/2, b/2, -b/2], [-h/2, h/2, h/2, -h/2, -h/2], '-', color='k', linewidth=1)    # Concrete section

# Coordinate axes
plt.plot([-1.2*b/2, 1.2*b/2], [0, 0], 'k', linewidth=0.3)
plt.plot([0, 0], [-1.2*h/2, 1.2*h/2], 'k', linewidth=0.3)
plt.annotate('$x$', (b/2+b/8, 0), verticalalignment='center')
plt.annotate('$y$', (0, h/2+h/8), horizontalalignment='center')

plt.plot([na_x0, na_x1], [na_y0, na_y1], color='k', linewidth=1)      # Plot neutral axis

plt.show()
