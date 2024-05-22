import numpy as np
import unyt as u

class R134Constants:
    """Experimental data and other constants for R134"""
    def __init__(self):
        assert (
            self.expt_liq_density.keys()
            == self.expt_vap_density.keys()
            == self.expt_Pvap.keys()
            == self.expt_Hvap.keys()
        )

    @property
    def molecular_weight(self):
        """Molecular weight of the molecule in g/mol"""
        return 102.0309

    @property
    def expt_Tc(self):
        """
        Critical temperature in K
        #Note: Using experimental properties of R134a since data for R134 is not available. These values will only be used as an initial guess for GEMC simulations.
        """
        return 374.21

    @property
    def expt_rhoc(self):
        """
        Critical density in kg/m^3
        #Note: Using experimental properties of R134a since data for R134 is not available. These values will only be used as an initial guess for GEMC simulations.
        """
        return 511.9

    @property
    def n_atoms(self):
        """Number of atoms in molecule"""
        return 8
    
    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "C(C(F)F)(F)F"
        
    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C",
            "sigma_F",
            "sigma_H",
            "epsilon_C",
            "epsilon_F",
            "epsilon_H",
        )

        return param_names
    
    @property
    def lit_param_set(self):
        lit_param_set = {
        }

        return lit_param_set

    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray([[2.0, 4.0], [2.0, 4.0], [1.5, 3.0],]) * u.Angstrom
            )  # C  # F  # H
            .in_units(u.nm)
            .value
        )

        bounds_epsilon = (
            (
                np.asarray(
                    [[10.0, 75.0], [15.0, 50.0], [2.0, 10.0],]
                )  # C  # F  # H
                * u.K
                * u.kb
            )
            .in_units("kJ/mol")
            .value
        )

        bounds = np.vstack((bounds_sigma, bounds_epsilon))

        return bounds

    @property
    def expt_liq_density(self):
        """Dictionary with experimental liquid density

        Temperature units K
        Density units kg/m**3
        #Note: Using experimental properties of R134a since data for R134 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_liq_density = {
            240: 1397.7,
            260: 1337.1,
            280: 1271.8,
            300: 1199.7,
            320: 1116.8,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        #Note: Using experimental properties of R134a since data for R134 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_vap_density = {
            240: 3.8367,
            260: 8.9052,
            280: 18.228,
            300: 34.193,
            320: 60.715,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        #Note: Using experimental properties of R134a since data for R134 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_Pvap = {
            240: (72.481 * u.kPa).to_value(u.bar),
            260: (176.84 * u.kPa).to_value(u.bar),
            280: (372.71 * u.kPa).to_value(u.bar),
            300: (702.82 * u.kPa).to_value(u.bar),
            320: (1216.6 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        #Note: Using experimental properties of R134a since data for R134 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_Hvap = {
            240: None,
            260: None,
            280: None,
            300: None,
            320: None,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        Use 2% as default when no data available
        """
        uncertainty = {
            "expt_liq_density": 0.02,
            "expt_vap_density": 0.02,
            "expt_Pvap": 0.02,
            "expt_Hvap": 0.02
        }
        return uncertainty
    
    @property
    def temperature_bounds(self):
        """Bounds on temperature in units of K"""

        lower_bound = np.min(list(self.expt_Pvap.keys()))
        upper_bound = np.max(list(self.expt_Pvap.keys()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def liq_density_bounds(self):
        """Bounds on liquid density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_liq_density.values()))
        upper_bound = np.max(list(self.expt_liq_density.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def vap_density_bounds(self):
        """Bounds on vapor density in units of kg/m^3"""

        lower_bound = np.min(list(self.expt_vap_density.values()))
        upper_bound = np.max(list(self.expt_vap_density.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def Pvap_bounds(self):
        """Bounds on vapor pressure in units of bar"""

        lower_bound = np.min(list(self.expt_Pvap.values()))
        upper_bound = np.max(list(self.expt_Pvap.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds

    @property
    def Hvap_bounds(self):
        """Bounds on enthaply of vaporization in units of kJ/kg"""

        lower_bound = np.min(list(self.expt_Hvap.values()))
        upper_bound = np.max(list(self.expt_Hvap.values()))
        bounds = np.asarray([lower_bound, upper_bound], dtype=np.float32)
        return bounds
