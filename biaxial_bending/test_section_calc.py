import unittest

import section_calc as sc


class TestSectionCalc(unittest.TestCase):

    def setUp(self):
        ''' 
        Setup a standard cross secion definition for testing in all methods. This avoids repitition of code in the
        beginning of each test method. 
        '''
        #================================================================================================
        # Tests based on Reinforced Concrete Mechanics and Design, Wight and MacGregor, Table 11-2
        #================================================================================================
        # Define concrete section
        self.x1 = [-8, 8, 8, -8]
        self.y1 = [8, 8, -8, -8]

        # Define rebar locations
        self.xr1 = [-5.6, 0,   5.6,  5.6,  5.6,  0,   -5.6, -5.6]
        self.yr1 = [5.6, 5.6, 5.6,  0,   -5.6, -5.6, -5.6,  0]

        # Netural axis location
        self.alpha_deg1 = 30
        self.na_y1 = -2

        # NOTE More tests could be setup with different variable names


    def test_compute_plastic_centroid(self):
        pass


    def test_compute_dist_to_na(self):

        #================================================================================================
        # Tests based on Reinforced Concrete Mechanics and Design, Wight and MacGregor, Table 11-2
        #================================================================================================
        # Run function and convert to rounded results before comparison
        dv, dr = sc.compute_dist_to_na(self.x1, self.y1, self.xr1, self.yr1, self.alpha_deg1, self.na_y1)
        dv = ['%.2f' % e for e in dv]   # Distance from neutral axis to concrete section vertices
        dr = ['%.2f' % e for e in dr]   # Distance from neutral axis to rebars

        # Perform tests for concrete vertices and rebar distances 
        self.assertEqual(dv, ['-12.66', '-4.66', '9.20', '1.20'])
        self.assertEqual(dr, ['-9.38', '-6.58', '-3.78', '1.07', '5.92', '3.12', '0.32', '-4.53'])

        # TODO Add more tests (edge cases)


    def test_compute_stress_block_geometry(self):
        #================================================================================================
        # Tests based on Reinforced Concrete Mechanics and Design, Wight and MacGregor, Table 11-2
        #================================================================================================
        # Distance from neutral axis to concrete section vertices
        dv = [-12.66, -4.66, 9.20, 1.20]
        # Distance from neutral axis to rebars
        dr = [-9.38, -6.58, -3.78, 1.07, 5.92, 3.12, 0.32, -4.53]
        # Run function and convert to rounded results before comparison
        x_sb, y_sb, Asb, sb_cog, c = sc.compute_stress_block_geometry(
            self.x1, self.y1, dv, dr, self.alpha_deg1, self.na_y1, lambda_=0.85)
        x_sb = ['%.2f' % e for e in x_sb]
        y_sb = ['%.2f' % e for e in y_sb]
        Asb = '%.0f' % Asb
        sb_cog = ['%.2f' % e for e in sb_cog]
        c = '%.2f' % c

        # Tests according to Reinforced Concrete Mechanics and Design, Wight and MacGregor, Figure 11-35
        self.assertEqual(x_sb, ['-8.00', '8.00', '8.00', '-8.00'])
        self.assertEqual(y_sb, ['-4.43', '4.81', '8.00', '8.00'])
        self.assertEqual(Asb, '%.0f' % (1/2*9.24*16 + 16*3.18))    
        # sb_cog
        # c


        pass



    def test_compute_rebar_strain(self):
        pass


if __name__ == '__main__':
    unittest.main()

