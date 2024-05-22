import numpy as np
import unyt as u

class R152Constants:
    """Experimental data and other constants for r152"""
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
        return 66.051

    @property
    def expt_Tc(self):
        """
        Critical temperature in K
        #Note: Using experimental properties of R152a since data for R152 is not available. These values will only be used as an initial guess for GEMC simulations.
        """
        return 386.41

    @property
    def expt_rhoc(self):
        """
        Critical density in kg/m^3
        #Note: Using experimental properties of R152a since data for R152 is not available. These values will only be used as an initial guess for GEMC simulations.
        """
        return 368

    @property
    def n_atoms(self):
        """Number of atoms in molecule"""
        return 8
    
    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "C(CF)F"
        
    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C1",
            "sigma_F1",
            "sigma_H1",
            "epsilon_C1",
            "epsilon_F1",
            "epsilon_H1",
        )

        return param_names
    
    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    [
                        [2.0,4.0], #[3.0, 4.0],  # C
                        [2.0,4.0], #[2.5, 3.5],  # F
                        [1.5, 3.0],  # H
                    ]
                )
                * u.Angstrom
            )
            .in_units(u.nm)
            .value
        )

        bounds_epsilon = (
            (
                np.asarray(
                    [
                        [10.0,75.0], #[20.0, 75.0],  # C C
                        [15.0,50.0], #[15.0, 40.0],  # F
                        [2.0, 10.0],  # H
                    ]
                )
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
        #Note: Using experimental properties of R152a since data for R152 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_liq_density = {
            260: 988.11,
            280: 943.43,
            300: 894.75,
            320: 840.12,
            340: 775.99,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        #Note: Using experimental properties of R152a since data for R152 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_vap_density = {
            260: 5.1996,
            280: 10.519,
            300: 19.497,
            320: 34.064,
            340: 57.642,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        #Note: Using experimental properties of R152a since data for R152 is not available. These values will only be used as an initial guess for GEMC simulations.
        """

        expt_Pvap = {
            260: (160.23 * u.kPa).to_value(u.bar),
            280: (335.36 * u.kPa).to_value(u.bar),
            300: (629.78 * u.kPa).to_value(u.bar),
            320: (1087.3 * u.kPa).to_value(u.bar),
            340: (1757.7 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            260: None,
            280: None,
            300: None,
            320: None,
            340: None,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: https://doi.org/10.1063/1.555979
        Uncertainties not mentioned as 2%
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
