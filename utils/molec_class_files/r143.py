import numpy as np
import unyt as u

class R143Constants:
    """Experimental data and other constants for R143"""
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
        return 84.04

    @property
    def expt_Tc(self):
        """Critical temperature in K"""
        return 429.8

    @property
    def expt_rhoc(self):
        """Critical density in kg/m^3"""
        return 469

    @property
    def n_atoms(self):
        """Number of atoms in molecule"""
        return 8
    
    @property
    def smiles_str(self):
        """Smiles string representation"""
        return "C(C(F)F)F"
        
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
            "sigma_H2",
            "epsilon_C1",
            "epsilon_C2",
            "epsilon_F1",
            "epsilon_F2",
            "epsilon_H1",
            "epsilon_H2",
        )

        return param_names
    
    @property
    def lit_param_set(self):
        """Adjustable parameter names"""

        lit_param_set = {
        }

        return lit_param_set

    @property
    def param_bounds(self):
        """Bounds on sigma and epsilon in units of nm and kJ/mol"""

        bounds_sigma = (
            (
                np.asarray(
                    [
                        [2.0,4.0], #[3.0, 4.0],  # C1
                        [2.0,4.0], #[3.0, 4.0],  # C2
                        [2.0, 4.0],  # F1
                        [2.0, 4.0],  # F2
                        [1.5, 3.0],  # H1
                        [1.5, 3.0],  # H1
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
                        [10.0,75.0], #[20.0, 70.0],  # C1
                        [10.0,75.0], #[20.0, 70.0],  # C2
                        [15.0,50.0], #[15.0, 40.0],  # F1
                        [15.0,50.0], #[15.0, 40.0],  # F2
                        [2.0, 10.0],  # H1
                        [2.0, 10.0],  # H1
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
        C.D. Holcomb, L.J. Van Poolen, Fluid Phase Equilibria (1994) https://doi.org/10.1016/0378-3812(94)80011-1.
        """

        expt_liq_density = {
            317: 1167.442,
            337: 1112.436,
            357: 1051.133,
            377: 980.341,
            397: 893.012,
        }

        return expt_liq_density

    @property
    def expt_vap_density(self):
        """Dictionary with experimental vapor density

        Temperature units K
        Density units kg/m**3
        C.D. Holcomb, L.J. Van Poolen, Fluid Phase Equilibria (1994) https://doi.org/10.1016/0378-3812(94)80011-1.
        """

        expt_vap_density = {
            317: 15.950,
            337: 27.281,
            357: 44.924,
            377: 72.080,
            397: 115.812,
        }

        return expt_vap_density

    @property
    def expt_Pvap(self):
        """Dictionary with experimental vapor pressure

        Temperature units K
        Vapor pressure units bar
        C.D. Holcomb, L.J. Van Poolen, Fluid Phase Equilibria (1994) https://doi.org/10.1016/0378-3812(94)80011-1.
        """

        expt_Pvap = {
            317: (429.027 * u.kPa).to_value(u.bar),
            337: (760.854 * u.kPa).to_value(u.bar),
            357: (1257.815 * u.kPa).to_value(u.bar),
            377: (1965.942 * u.kPa).to_value(u.bar),
            397: (2935.715 * u.kPa).to_value(u.bar),
        }

        return expt_Pvap

    @property
    def expt_Hvap(self):
        """Dictionary with experimental enthalpy of vaporization

        Temperature units K
        Enthalpy of vaporization units kJ/kg
        """

        expt_Hvap = {
            317: None,
            337: None,
            357: None,
            377: None,
            397: None,
        }

        return expt_Hvap

    @property
    def uncertainties(self):
        """
        Dictionary with uncertainty for each calculation
        Use 2% as default when no data available
        """
        uncertainty = {
            "expt_liq_density": 0.0041,
            "expt_vap_density": 0.0041,
            "expt_Pvap": 0.0007,
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
