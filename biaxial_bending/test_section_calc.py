import unittest

import section_calc as sc


class TestSectionCalc(unittest.TestCase):

    def set_up(self):
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


    def test_elastic_centroid(self):
        #================================================================================================
        # Example 9.1 from 'Reinforced Concrete Mechanics and Design', Wight and MacGregor
        #================================================================================================
        # Note: Centroid distance found is measured from top of section in the example, but
        # the coordinates below are changed so they match measurements from the bootom of 
        # the section.

        # UNCRACKED STATE
        b = 12
        h = 24
        d = [0.875] * 6      # Rebar diameters [in] (No. 7 bars)
        c = 2.5
        Ec = 1   # [psi]
        Es = 8     # [psi]

        x = [0, b, b, 0]
        y = [0, 0, h, h]
        xr = [2, 4, 8, 10, 2, 10]
        yr = [c, c, c, c, h-c, h-c]

        xel, yel = sc.elastic_centroid(x, y, xr, yr, d, Ec=Ec, Es=Es)

        # Convert y-result to rounded string and perform comparison
        self.assertEqual('%.1f' % yel, '11.7')
        self.assertEqual(xel, b/2)

        # CRACKED STATE


    def test_compute_plastic_centroid(self):
        pass


    def test_Itx(self):

        #================================================================================================
        # Example 9.1 from 'Reinforced Concrete Mechanics and Design', Wight and MacGregor
        #================================================================================================
        
        b = 12
        h = 24
        d = [0.875] * 6      # Rebar diameters [in] (No. 7 bars)
        c = 2.5
        Ec = 3.6*10**6   # [psi]
        Es = 29*10**6     # [psi]

        x = [0, b, b, 0]
        y = [0, 0, h, h]
        xr = [2, 4, 8, 10, 2, 10]
        yr = [c, c, c, c, h-c, h-c]
 
        # UNCRACKED STATE (All concrete is effective)

        # Note: The example in the book uses rough numerical round-offs. However, the final
        # result in the book comes out as I=16100[in^2], while the 'exact' result is
        # I=16101[in^2]. The computations in this test uses many decimal points.
        yc_uncracked = 11.7
        Ix_uncracked = sc.Itx(yc_uncracked, x, y, xr, yr, d, Ec=Ec, Es=Es)
        self.assertEqual('%.0f' % Ix_uncracked, '16101')

        # CRACKED STATE (Concrete in tension is ineffective)

        # Centroid of cracked sectionAyc 

        yc_cracked = 24 - 6.51

        # Coordinates for stress block vertices after cracking 
        x_cracked = [0, b, b, 0]
        y_cracked = [yc_cracked, yc_cracked, h, h]

        Ix_cracked = sc.Itx(yc_cracked, x_cracked, y_cracked, xr, yr, d, Ec=Ec, Es=Es)
        self.assertEqual('%.0f' % Ix_cracked, '5594')


    def test_compute_dist_to_na(self):

        # #================================================================================================
        # # Tests based on Reinforced Concrete Mechanics and Design, Wight and MacGregor, Table 11-2
        # #================================================================================================
        # # Run function and convert results to rounded strings before comparison
        # dv, dr = sc.compute_dist_to_na(self.x1, self.y1, self.xr1, self.yr1, self.alpha_deg1, self.na_y1)
        # dv_r = ['%.2f' % e for e in dv]   # Distance from neutral axis to concrete section vertices
        # dr_r = ['%.2f' % e for e in dr]   # Distance from neutral axis to rebars

        # # Perform tests for concrete vertices and rebar distances 
        # self.assertEqual(dv_r, ['-12.66', '-4.66', '9.20', '1.20'])
        # self.assertEqual(dr_r, ['-9.38', '-6.58', '-3.78', '1.07', '5.92', '3.12', '0.32', '-4.53'])

        # TODO Add more tests (edge cases)
        pass


    def test_compute_stress_block_geometry(self):
        # #================================================================================================
        # # Tests based on Reinforced Concrete Mechanics and Design, Wight and MacGregor, Table 11-2
        # #================================================================================================
        # # Distance from neutral axis to concrete section vertices
        # dv = [-12.66, -4.66, 9.20, 1.20]
        # # Distance from neutral axis to rebars
        # dr = [-9.38, -6.58, -3.78, 1.07, 5.92, 3.12, 0.32, -4.53]
        # # Run function and convert to rounded results before comparison
        # x_sb, y_sb, Asb, sb_cog, c = sc.stress_block_geometry(
        #     self.x1, self.y1, dv, dr, self.alpha_deg1, self.na_y1, lambda_=0.85)
        # x_sb = ['%.2f' % e for e in x_sb]
        # y_sb = ['%.2f' % e for e in y_sb]
        # Asb = '%.0f' % Asb
        # sb_cog = ['%.2f' % e for e in sb_cog]
        # c = '%.2f' % c

        # # Tests according to Reinforced Concrete Mechanics and Design, Wight and MacGregor, Figure 11-35
        # self.assertEqual(x_sb, ['-8.00', '8.00', '8.00', '-8.00'])
        # self.assertEqual(y_sb, ['-4.43', '4.81', '8.00', '8.00'])
        # self.assertEqual(Asb, '%.0f' % (1/2*9.24*16 + 16*3.18))    
        # # sb_cog
        # # c
        pass

        



    def test_compute_rebar_strain(self):
        pass


if __name__ == '__main__':
    unittest.main()

