import re

def get_points(string):
    '''
    Find all 2D points of the format '(x, y)' in the input string.
    All characters are valid as delimeters between correctly typed points.

    INPUT
        string  (str)   -   String to find coordinates within

    OUTPUT
        x       (list)  -   List of x-coordinates (each entry being a float)
        y       (list)  -   List of y-coordinates (each entry being a float)
        c       (list)  -   List of found points represented by tuples, i.e. [(x1, y1), (x2, y2), ..., (xn, yn)]
    '''

    if len(string) == 0:
        x = []
        y = []
        c = []
        return x, y, c

    else:
        # Create regular expression for syntax of corretly typed point
        pnd = '[+-]?\d+(?:\.\d+)?'      # Regex for searching for positives, negatives and/or decimals
        regex = re.compile(r'\(\s*' + str(pnd) + '\s*,\s*' + str(pnd) + '\s*\)')
        # NOTE Should user be allowed to type e.g. letters or special characters=?/! in the text field?

        # Find all coordinates and put them in a list of strings
        c = re.findall(regex, string)

        # Turn list of strings into list of coordinate tuples
        c = [eval(coord) for coord in c]

        # Extract x- and y-coordinates from each coordinate tuple
        x = [coord[0] for coord in c]
        y = [coord[1] for coord in c]

        return x, y, c

s = '(1, 2), (0, 0.1), (3, -4), (-1.15, 6) (   7,    8), (9, (12.1321, 17.1)'
x, y, c = get_points(s)

print(x)
print(y)
print(c)
