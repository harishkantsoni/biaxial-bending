import unittest

import geometry


class TestGeometry(unittest.TestCase):

    def test_line_polygon_intersections(self):

        # Line
        angle = 30
        y_intersect = -2
        
        # Define polygon (rectangle in this case)
        x_vertex = [-8, 8, 8, -8]
        y_vertex = [8, 8, -8, -8]

        # Run function and convert round results before comparison
        x_int, y_int = geometry.line_polygon_collisions(angle, y_intersect, x_vertex, y_vertex)

        x_int = ['%.2f' % c for c in x_int]
        y_int = ['%.2f' % c for c in y_int]

        # Perform tests for line polygon collisions
        self.assertEqual(x_int, ['8.00', '-8.00'])
        self.assertEqual(y_int, ['2.62', '-6.62'])

    def test_polygon_area():
        pass


if __name__ == '__main__':
    unittest.main()
