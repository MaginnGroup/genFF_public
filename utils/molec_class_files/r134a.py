import numpy as np
import unyt as u

class R134aConstants:
    """Experimental data and other constants for r134a"""
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
        """Critical temperature in K"""
        return 374.21

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 511.9

    @property
    def n_params(self):
        """Number of adjustable parameters"""
        return len(self.param_names)

    @property
    def param_names(self):
        """Adjustable parameter names"""

        param_names = (
            "sigma_C1",
            "sigma_C2",
            "sigma_F1",
            "sigma_F2",
            "sigma_H1",
            "epsilon_C1",
            "epsilon_C2",
            "epsilon_F1",
            "epsilon_F2",
            "epsilon_H1",
        )

        return param_names
    
    @property
    def lit_param_set(self):
        """Adjustable parameter names"""

        lit_param_set = {
            "sigma_C1":3.745,
            "sigma_C2": 3.754,
            "sigma_F1":2.982,
            "sigma_F2":2.607,
            "sigma_H1":2.237,
            "epsilon_C1":20.73,
            "epsilon_C2":72.61,
            "epsilon_F1":23.13,
            "epsilon_F2":39.98,
            "epsilon_H1":2.55,
        }

        return lit_param_set

    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    [
                        [2.0,4.0], #[3.0, 4.0],  # C
                        [2.0,4.0], #[3.0, 4.0],  # C
                        [2.0,4.0], #[2.5, 3.5],  # F
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
                        [10.0,75.0], #[20.0, 75.0],  # C
                        [10.0,75.0], #[20.0, 75.0],  # C
                        [15.0,50.0], #[15.0, 40.0],  # F
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
        """

        expt_Hvap = {
            240: 221.55,
            260: 208.20,
            280: 193.28,
            300: 176.08,
            320: 155.48,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        from: https://doi.org/10.1063/1.555958
        """
        uncertainty = {
            "expt_liq_density": 0.0005,
            "expt_vap_density": 0.0005,
            "expt_Pvap": 0.0002,
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
