import numpy as np
import unyt as u

class R116Constants:
    """Experimental data and other constants for R116"""
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
        return 138.01182

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 293.03

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 613.32

    @property
    def n_atoms(self):
        """Number of atoms in molecule"""
        return 8

    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "C(C(F)(F)F)(F)(F)F"

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
            "epsilon_C1",
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
                        [2.0, 4.0],  # C
                        [2.0, 4.0], #[2.5, 3.5],  # F
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
                        [10.0, 75.0],  # C
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
            190: 1627,
            210: 1536.6,
            230: 1435.5,
            250: 1317,
            270: 1164,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        """

        expt_vap_density = {
            190: 6.9451,
            210: 18.128,
            230: 40.175,
            250: 80.857,
            270: 158.47,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        """

        expt_Pvap = {
            190: (76.558 * u.kPa).to_value(u.bar),
            210: (211.94 * u.kPa).to_value(u.bar),
            230: (481.5 * u.kPa).to_value(u.bar),
            250: (950.2 * u.kPa).to_value(u.bar),
            270: (1695.2 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            190: 119.13,
            210: 110.15,
            230: 99.46,
            250: 85.98,
            270: 67.3,
        }

        return expt_Hvap
    
    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: https://doi.org/10.1021/je050186n
        Note: Hvap is set at 2% because a value was not given in the reference provided
        """
        uncertainty = {
            "expt_liq_density": 0.002,
            "expt_vap_density": 0.002,
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
