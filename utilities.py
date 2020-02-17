import CoolProp as CP
from math import isclose

class FluidStation:

    """
    Incompressible FluidStation class used to perform all fluid property 
    calculations.

    Uses CoolProp as an interface for REFPROP calculations.
    ----------
    t     : float
        Static temperature
    p     : float
        Static pressure
    r_gas : float
        Specific gas constant.
    comp  : dict
        Composition specified by molecule name and volume fraction.
    mu    : float
        Dynamic viscosity
    cp    : float
        Specific heat capacity
    k     : float
        Thermal conductivity
    pr    : float
        Prandtl number
    """

    def __init__(self, t, p, fluid):
        """Initialization of incompressible fluid station class. 
        All calculations are done assuming total and static properties 
        are the same.

        Inputs
        ----------
        t : float
            Static temperature
        p : float
            Static pressure
        fluid : str
            String indicating the type of fluid present in the station
        """

        self._t = t
        self._p = p

        if fluid == 'Air':
            self._comps_str = 'Nitrogen&Oxygen&Argon'
            self._comps_y = [0.7812,0.2096,0.0092]
        else:
            self._comps_str = fluid[0]
            self._comps_y = fluid[1]

        self._cp_state =  self.state_gen(self._comps_str,self._comps_y)
        self.state_update('PT',p,t)

        # Calculate properties from CP
        self._r = self._cp_state.gas_constant() /self._cp_state.molar_mass() 
        self._cp = self._cp_state.cpmass()
        self._h =  self._cp_state.hmass()
        self._s =  self._cp_state.smass()
        
        # Avoid error with transport property calcs when water contents are above 5%
        try:
            self._mu = self._cp_state.viscosity()
        except:
            dummy_comp_y = [0.768, 0.128, 0.054, 0.05] # Dummy flue gas composition
            self._cp_state =  self.state_gen(self._comps_str,dummy_comp_y)
            self.state_update('PT',p,t)
            self._mu = self._cp_state.viscosity()
            
        self._k =  self._cp_state.conductivity()

    def state_gen(self,comps_cp,fractions,props_db='REFPROP'):

        if props_db not in ['HEOS','REFPROP']:
            print('Unavailable fluid DB selected. HEOS will be used')
            props_db = 'REFPROP'

        state = CP.AbstractState(props_db,comps_cp)
        state.set_mole_fractions(fractions) 
       # state.build_phase_envelope('dummy_variable')

        return state

    def state_update(self,input_pair,in1,in2):

        ip_dict = {
            'PT':CP.PT_INPUTS,
            'PS':CP.PSmass_INPUTS,
            'HP':CP.HmassP_INPUTS,
        }
        
        if input_pair not in ip_dict.keys():
            raise ValueError('Selected input pair is not available')

        self._cp_state.update(ip_dict[input_pair],in1,in2)

        return None

    def copy(self):

        d_copy = {
            't':self.t,
            'p':self.p,
            'fluid':self.fluid
        }
        return d_copy

    @property
    def t(self):
        return self._cp_state.T()

    @property
    def p(self):
        return self._cp_state.p()

    @property
    def fluid(self):
        return [self._comps_str, self._comps_y]

    @property
    def r_gas(self):
        return self._cp_state.gas_constant() /self._cp_state.molar_mass() 

    @property
    def cp(self):
        return self._cp_state.cpmass()

    @property
    def h(self):
        return self._cp_state.hmass()

    @property
    def s(self):
        return self._cp_state.smass()

    @property
    def mu(self):
        return self._cp_state.viscosity()

    @property
    def k(self):
        return self._cp_state.conductivity()

    @property
    def rho(self):
        return self._p / (self._r * self._t)

    @property
    def pr(self):
        _cp = self._cp_state.cpmass()
        _mu = self._cp_state.viscosity()
        _k = self._cp_state.conductivity()
        return  _cp * _mu / _k
