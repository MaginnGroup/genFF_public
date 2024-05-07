import numpy as np
import unyt as u

class R14Constants:
    """Experimental data and other constants for R14"""
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
        return 88.0043

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 227.51

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 625.66161

    @property
    def n_atoms(self):
        """Number of adjustable parameters"""
        return 5

    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "C(F)(F)(F)F"
    
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
    def lit_param_set(self):
        """Adjustable parameter names"""

        lit_param_set = {
            "sigma_C1":3.4895482012148435,
            "sigma_F1":2.917216063390504,
            "epsilon_C1":36.246581269369244,
            "epsilon_F1":29.068536341766947,
        }

        return lit_param_set

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
            130: 1681.5,
            150: 1576.8,
            170: 1461.2,
            190: 1325.3,
            210: 1142.5,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        """

        expt_vap_density = {
            130: 2.5091,
            150: 10.432,
            170: 30.641,
            190: 74.361,
            210: 169.76,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        """

        expt_Pvap = {
            130: (30.392 * u.kPa).to_value(u.bar),
            150: (141.20 * u.kPa).to_value(u.bar),
            170: (440.54 * u.kPa).to_value(u.bar),
            190: (1066.3 * u.kPa).to_value(u.bar),
            210: (2186.4 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            130: 142.00,
            150: 131.72,
            170: 118.77,
            190: 101.50,
            210: 75.63,
        }

        return expt_Hvap
    
    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: Platzer, B.; Polt, A.; Maurer, G. Thermophysical Properties of Refrigerants; Springer-Verlag: Berlin, 1990.
        Note: Hvap is set at 1% because a value was not given in the reference provided
        """
        uncertainty = {
            "expt_liq_density": 0.0164125,
            "expt_vap_density": 0.0034,
            "expt_Pvap": 0.001038,
            "expt_Hvap": 0.01
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
