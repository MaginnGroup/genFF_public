import numpy as np
import unyt as u

class R23Constants:
    """Experimental data and other constants for R23"""
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
        return 70.014

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 299.29

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 526.5

    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C1",
            "sigma_H1",
            "sigma_F1",
            "epsilon_C1",
            "epsilon_H1",
            "epsilon_F1",
        )

        return param_names
    
    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    [
                        [2.0, 4.0], #[3.0, 4.0],  # C
                        [1.5, 3.0],  # H
                        [2, 4.0],  # F
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
                        [10.0,75.0], #[20.0, 75.0],  # C
                        [2.0, 10.0],  # H
                        [15.0, 50.0],  # F
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
        """

        expt_liq_density = {
            190: 1449.9,
            210: 1371.2,
            230: 1284.4,
            250: 1184.2,
            270: 1058.8,
        }

        return expt_liq_density
    
    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        """

        expt_vap_density = {
            190: 4.3834,
            210: 11.761,
            230: 26.64,
            250: 54.416,
            270: 106.58,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        """

        expt_Pvap = {
            190: (94.862 * u.kPa).to_value(u.bar),
            210: (269.91 * u.kPa).to_value(u.bar),
            230: (627.55 * u.kPa).to_value(u.bar),
            250: (1262.8 * u.kPa).to_value(u.bar),
            270: (2288.7 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            190: 240.283,
            210: 222.94,
            230: 202.6,
            250: 177.51,
            270: 144.14,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: https://doi.org/10.1063/1.1559671 
        Hvap not mentioned, put 2%
        """
        uncertainty = {
            "expt_liq_density": 0.001,
            "expt_vap_density": 0.001,
            "expt_Pvap": 0.002,
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
