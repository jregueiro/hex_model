import scipy.special as sp
from math import pi, sqrt, log
import numpy as np
from utilities import FluidStation

"""
References used in this module:
    [1] Hesselgreaves, J. E. (2017). 
        Compact heat exchangers: Selection, design, and operation.
    [2] Manglik, R. M., & Bergles, A. E. (1995). 
        Heat transfer and pressure drop correlations 
        for the rectangular offset strip fin compact heat exchanger. 
        Experimental Thermal and Fluid Science, 10(2), 171-180
    [3] Awad, M., & Muzychka, Y. S. (2011). 
        Models for pressure drop and heat transfer in air 
        cooled compact wavy fin heat exchangers. 
        Journal of Enhanced Heat Transfer, 18(3).

"""


class PlateFin:

    """
    Base secondary surface class that contains methods for: 
       geometric calculations
       j/f performance

       Nomenclature:
       s - fin spacing
    """

    def __init__(self, b, p_f, t_f, l_f, t_w):
        self._b = b
        self._p_f = p_f
        self._t_f = t_f
        self._l_f = l_f
        self._t_w = t_w

    @property
    def h_i(self):
        return self._b - self._t_f

    @property
    def s(self):
        return self._p_f - self._t_f

    @property
    def a_o_cell(self):
        return self.s * self.h_i

    @property
    def a_cell(self):
        return 2*(self.s*self._l_f + self.h_i*self._l_f + self.h_i*self._t_f)\
            + self.s*self._t_f

    @property
    def d_h(self):
        return 4 * self.a_o_cell * self._l_f / self.a_cell

    @property
    def sigma(self):
        return (self.s*(self._b-self._t_f)) /\
            ((self.s-self._t_f*(self._b+self._t_w)))

    @property
    def alpha(self):
        return self.s/self.h_i

    @property
    def delta(self):
        return self._t_f/self._l_f

    @property
    def gamma(self):
        return self._t_f/self.s


class OSF(PlateFin):

    def j_re(self, re, correl='ManglikBergles'):
        """j correlations from Ref. [3]
        """

        if correl == 'ManglikBergles':
            self.correl = correl
            x1, x2, x3 = self.alpha, self.delta, self.gamma
            e0 = [-0.5403, -0.1541, 0.1499, -0.0678]
            e1 = [1.3400,  0.5040, 0.4560, -1.0550]
            a1, b1 = 0.6522, 5.269e-5
            j_1 = a1 * re**e0[0] * x1**e0[1] * x2**e0[2] * x3**e0[3]
            j_2 = (1 + b1 * re**e1[0] * x1**e1[1] * x2**e1[2] * x3**e1[3])**0.1
            return j_1 * j_2
        else:
            raise Exception(
                'This j correlation is not available for the selected geometry')

    def f_re(self, re, correl='ManglikBergles'):
        """f correlations from Ref. [3]
        """
        if correl == 'ManglikBergles':
            self.correl = correl
            x1, x2, x3 = self.alpha, self.delta, self.gamma
            e0 = [-0.7422, -0.1856, 0.3053, -0.2659]
            e1 = [4.4290,  0.9200, 3.7670,  0.2360]
            a1, b1 = 9.6243, 7.669e-8
            f_1 = a1 * re**e0[0] * x1**e0[1] * x2**e0[2] * x3**e0[3]
            f_2 = (1 + b1 * re**e1[0] * x1**e1[1] * x2**e1[2] * x3**e1[3])**0.1
            return f_1 * f_2
        else:
            raise Exception(
                'This f correlation is not available for the selected geometry')


class WavyFin(PlateFin):

    def __init__(self, b, p_f, t_f, a_wave, p_wave, t_w):
        self._b = b
        self._p_f = p_f
        self._t_f = t_f
        self._a_wave = a_wave
        self._wl_wave = p_wave
        self._t_w = t_w

    @property
    def gamma(self):
        return 2*self._a_wave/self._wl_wave

    @property
    def l_f(self):
        f_ellipe = sp.ellipe((self.gamma * pi) /
                             sqrt(1 + self.gamma**2 * pi**2))
        return 2*self._wl_wave*sqrt(1 + self.gamma**2 * pi**2)*f_ellipe/pi

    @property
    def d_h(self):
        return (2*self.s)/(self.alpha+1)

    def j_re(self, re, pr, correl='MuzychkaAwad'):
        """j correlations from Ref. [2]
        """
        if correl == 'MuzychkaAwad':
            # Equation coefficients
            jw_c0, j_lbl_c = 7.541, 0.664
            jw_c_l = [-0.548, 2.702,  -5.119, 4.970, - 2.610, 0]

            # j calculation
            j_alpha = np.poly1d(jw_c_l)
            j_wavy = jw_c0 / (re * pr**(1 / 3)) * (1 + j_alpha(self.alpha))
            j_lbl = j_lbl_c / sqrt(re) * sqrt(self.d_h / (0.5*self.l_f))
            return (j_wavy**5 + j_lbl**5)**0.2
        else:
            raise Exception(
                'This j correlation is not available for the selected geometry')

    def f_re(self, re, pr, correl='MuzychkaAwad'):
        """f correlations from Ref. [2]
        """
        if correl == 'MuzychkaAwad':
            # Equation coefficients
            f_comp_1_c, fRe_c = 48, 3.44
            f_comp_3_coeffs = [-0.2537, 0.9564, -1.7012, 1.9467, -1.3553, 0]
            g_pi_fac = sqrt(1 + self.gamma**2 * pi**2)

            # f calculation
            f_comp_1 = f_comp_1_c * g_pi_fac / pi
            f_comp_2 = sp.ellipe((self.gamma * pi) / g_pi_fac)
            f_alpha_3 = np.poly1d(f_comp_3_coeffs)
            f_comp_3 = (1 + f_alpha_3(self.alpha))

            fRe_wavy = f_comp_1 * f_comp_2 * f_comp_3
            l_plus = self.l_f / (self.d_h * re)
            fRe_app = fRe_c / sqrt(l_plus)
            fRe = sqrt(fRe_wavy**2 + fRe_app**2)
            f = fRe / re
            return f
        else:
            raise Exception(
                'This f correlation is not available for the selected geometry')


class PrimarySurface:

    def __init__(self, name, age):
        self.name = name
        self.age = age


class ExchangerSide:

    def __init__(self, fs_i, fs_o, mdot, surf):

        # Assign inlet variables
        self._t_i = fs_i.t
        self._p_i = fs_i.p
        self._rho_i = fs_i.rho

        # Assign outlet variables
        self._t_o = fs_o.t
        self._p_o = fs_o.p
        self._rho_o = fs_o.rho

        # Calculate mean fluid properties
        self.mdot = mdot
        self.rho_m = np.mean([fs_i.rho, fs_o.rho])
        self.cp_m = np.mean([fs_i.cp, fs_o.cp])
        self.mu_m = np.mean([fs_i.mu, fs_o.mu])
        self.pr_m = np.mean([fs_i.pr, fs_o.pr])

        self.d_h = surf.d_h
        self.surf = surf

    @property
    def dp(self):
        return self._p_i - self._p_o

    @property
    def a_c(self):
        return self.mdot/self.g

    @property
    def re(self):
        return self.g*self.d_h/self.mu_m

    @property
    def j(self):
        if self.surf.__class__.__name__ == 'WavyFin':
            j = self.surf.j_re(self.re, self.pr_m)
        else:
            j = self.surf.j_re(self.re)
        return j

    @property
    def f(self):
        if self.surf.__class__.__name__ == 'WavyFin':
            f = self.surf.f_re(self.re, self.pr_m)
        else:
            f = self.surf.f_re(self.re)
        return f

    def g_jf_set(self, jf_rat, n):
        """
        Set G using j/f ratio input.
        Core mass velocity equation
        Reference [1] - Eq. 4.13 
        """
        g_sq = (2*self.rho_m*self.dp)*(jf_rat/(self.pr_m**0.666*n))

        self.g = g_sq**0.5

    def l_jn_set(self, n):
        """
        Set L using j factor and n
        Core mass velocity equation
        Reference [1] - Eq. 4.25
        """
        self.l = self.d_h*self.pr_m**0.666*n/(4*self.j)

    # ntu
    # area_fin
    # area_plate
    # area_total
    # frontal area
    # free flow area
    # inlet fluid
    # average fluid
    # outlet fluid
    # passage width
    # number of passages


class Exchanger:

    def __init__(self, side_1, side_2, eps, flow_arr):
        """Init routine is based on scoping size methodology in Ref. [1]-pg 333 

        After the init routine is done, a stream length is obtained for each
        side which results in the effectiveness and pressure drop.
        """

        self._s_1 = side_1
        self._s_2 = side_2
        self._eps_tar = eps
        self._flow_arr = flow_arr

        # Init heat capacity ratio with inlet specific heats
        c_1 = side_1.mdot * side_1.cp_m
        c_2 = side_2.mdot * side_2.cp_m
        self.c_star_init = c_1/c_2
        ntu_init = self.ntu_e(self.c_star_init, eps)
        n_1_init, n_2_init = 2*ntu_init, 2*ntu_init   # Assume a balanced HEX

        # Set G and Re from j/f ratio assumption and core mass velocity for each side
        # Get L from new calculations of Re
        jf_rat = 0.25

        for (side, n) in zip([self._s_1, self._s_2], [n_1_init, n_2_init]):
            # Set G
            side.g_jf_set(jf_rat, n)
            # Set length
            side.l_jn_set(n)

    def ntu_e(self, c_star, eps):
        """
        Inverse NTU-eps relationship for CF and XF arrangements
        Reference [1] - Table 7.2 
        XF relationship assumes unmixed C_min stream
        """
        if self._flow_arr == 'CF':
            ntu = (1/(c_star-1))*log((eps-1)/(c_star*eps-1))
        elif self._flow_arr == 'XF':
            ntu = -log(1+(1/c_star)*log(1-c_star*eps))
        else:
            raise Exception('Incorrect value of the flow arangement string')
        return ntu

    # Flow arrangement
    # NTU
    # Overall U
    #

if __name__ == "__main__":
    """Parameter         Temperature [K]  Pressure [bar]
    ====================================================
       Inlet                293.15           1.0133
       Compressor Inlet     293.15           0.9633
       HEX HP Inlet         471.99           3.853
       Combustor Inlet      861.18           3.7181
       Turbine Inlet        1200             3.6441
       HEX LP Inlet         940.93           1.0703
       HEX LP Outlet        575.89           1.0382
       Exhaust              575.89           1.0133

    """
    t_1_in, p_1_in, fluid_1 = 470.00, 3.85e5, 'Air'
    t_2_in, p_2_in, fluid_2 = 940.00, 1.07e5, 'Air'
    mdot_1, mdot_2 = 0.258, 0.260

    # Recuperator performance
    eps = 0.83
    dpr_1 = 0.03
    dpr_2 = 0.035

    #Define surfaces
    surf_1 = OSF(0.00191,0.001053,0.000076,0.00282,0.0003)
    surf_2 = OSF(0.00254,0.001053,0.000076,0.00282,0.0003)

    # Define inlet Fluid Stations
    fs_1_in = FluidStation(t_1_in,p_1_in,fluid_1)
    fs_2_in = FluidStation(t_2_in,p_2_in,fluid_2)

    # Define outlet Fluid Stations
    c_rat = fs_1_in.cp/fs_2_in.cp
    t_1_out = t_1_in+eps*(t_2_in-t_1_in)
    t_2_out = t_2_in-(eps/c_rat)*(t_2_in-t_1_in)

    p_1_out = p_1_in*(1-dpr_1)
    p_2_out = p_2_in*(1-dpr_2)
    
    fs_1_out = FluidStation(t_1_out,p_1_out,fluid_1)
    fs_2_out = FluidStation(t_2_out,p_2_out,fluid_2)

    # Define sides
    side_1 = ExchangerSide(fs_1_in,fs_1_out,mdot_1,surf_1)
    side_2 = ExchangerSide(fs_2_in,fs_2_out,mdot_2,surf_2)

    # Preliminary Exchanger
    hexchanger = Exchanger(side_1,side_2,eps,'CF')

    

# class FlexColdFin:
#     def __init__(self):
#         self.fin_type = 'wavy_fin'
#         self.a_sin = 0.00126
#         self.l_sin = 0.009525
#         self.pitch = 0.000591
#         self.height = 0.00254
#         self.th = 0.000076
#         self.k = 18.0

# class FlexHotFin:
#     def __init__(self):
#         self.fin_type = 'wavy_fin'
#         self.a_sin = 0.00126
#         self.l_sin = 0.009525
#         self.pitch = 0.000847
#         self.height = 0.0047
#         self.th = 0.000076
#         self.k = 18.0

# class OffsetColdValidation:
#     def __init__(self):
#         self.pitch = 0.001053
#         self.height = 0.00191
#         self.th = 0.00010
#         self.length = 0.00282

# class OffsetHotValidation:
#     def __init__(self):
#         self.pitch = 0.001053
#         self.height = 0.00254
#         self.th = 0.00010
#         self.length = 0.00282        