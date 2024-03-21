from .r14 import R14Constants
from .r32 import R32Constants
from .r50 import R50Constants
from .r125 import R125Constants
from .r134a import R134aConstants
from .r143a import R143aConstants
from .r170 import R170Constants
import numpy as np
import unyt as u

class Atom_Types:
    """
    Base class for atom typing schemes

    Methods
    -------
    __init__(at_bounds, at_names, molec_map_dicts)
    get_transformation_matrix(self, molec_key)
    check_for_duplicates(self)
    """

    def __init__(self, at_bounds, at_names, molec_map_dicts):
        """
        Initialization Method:

        Parameters
        ----------
        at_bounds: array, The bounds of the new atom type scheme
        at_names: list, The names of the new atom types. Should correspond to each element in at_bounds.shape[1]
        molec_map_dicts: dict, The dictionary of molecule property class for the old atom types
        """
        assert isinstance(at_bounds, np.ndarray), "at_bounds must be an np.ndarray"
        assert isinstance(at_names, list), "at_names must be a list"
        assert isinstance(molec_map_dicts, dict), "molec_map_dicts must be a dictionary"
        assert all(isinstance(name, str) for name in at_names) == True, "all at_names must be string"
        assert len(at_names) == at_bounds.shape[0], "at_bounds must have one column for each name in at_names"

        self.at_bounds = at_bounds
        self.at_names = at_names
        self.molec_map_dicts = molec_map_dicts
        self.at_matrices = {}

    def scale_bounds(self):
        """
        Scales bounds to units of nm and kj/mol
        """
        #Get upper and lower bounds seperately
        bounds_list = [self.at_bounds[:,x] for x in range(self.at_bounds.shape[1])]
        at_bounds_nm_kjmol = np.zeros(self.at_bounds.shape)
        #Get Midpoint of bounds
        midpoint = len(bounds_list[0]) // 2
        #Loop over upper and lower bounds
        for i in range(len(bounds_list)):
            #Create scaled list of upper and lower bounds for sigmas and epsilons
            sigmas = [float((x * u.Angstrom).in_units(u.nm).value) for x in bounds_list[i][:midpoint]]
            epsilons = [float((x * u.K * u.kb).in_units("kJ/mol").value) for x in bounds_list[i][midpoint:]]
            # Combine the results and add to array
            new_bound = np.array(sigmas + epsilons)
            at_bounds_nm_kjmol[:,i] = new_bound

        self.at_bounds_nm_kjmol = at_bounds_nm_kjmol


    def get_transformation_matrix(self, molec_key):
        """
        Creates transformation matrix between new and old atom types

        Parameters:
        -----------
        molec_key: str, a key from the molec_map_dicts dictionary
        
        Returns:
        --------
        at_matrix: array, The transformation matrix from new to old atom types
        """
        assert molec_key in self.molec_map_dicts
        #If you already have this matrix, use it. Otherwise generate it
        if not molec_key in self.at_matrices:
            #Get mapping for specific molecule
            map_dict = self.molec_map_dicts[molec_key]
            #Create a matrix based on the keys and map dict length
            at_matrix = np.zeros((len(self.at_names), len(map_dict)))
            # Fill at_matrix with ones or zeros based on the presence of keys
            for i, value in enumerate(self.at_names):
                if value in map_dict.values():
                    at_matrix[i, list(map_dict.values()).index(value)] = 1
            #Add matrix to self dictionary
            self.at_matrices[molec_key] = at_matrix
        else:
            at_matrix = self.at_matrices[molec_key]
        return at_matrix
    
    def check_for_duplicates(self):
        """
        Checks for duplicates in at_matrix
        """
        arr_list = list(self.at_matrices.values())
        tuple_list = [tuple(map(tuple, arr)) for arr in arr_list]
        len(tuple_list) != len(set(tuple_list))

        return len(tuple_list) != len(set(tuple_list))
    

class AT_Scheme_7(Atom_Types):
    """
    Class for Atom Type Scheme 7

    Methods
    -------
    __init__(self)
    """
    def __init__(self):
        #Get Bounds
        at_param_bounds_l = [2, 2, 1.5, 2, 2, 2, 10, 10,  2, 15, 15, 15] #Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4, 4,   3, 4, 4, 4, 75, 75, 10, 50, 50, 50]
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_7"
        #Get Names
        at_keys = ["sigma_C1", "sigma_C2", "sigma_H1","sigma_F_H2","sigma_F_H1","sigma_F_Hx",
           "epsilon_C1", "epsilon_C2", "epsilon_H1","epsilon_F_H2","epsilon_F_H1","epsilon_F_Hx"]

        #Create a file that maps param names (keys) to at_param names for atom type 7 (values) for each molecule
        r14_map_dict = {"sigma_C1": "sigma_C1",
                    "sigma_F1": "sigma_F_Hx",
                    "epsilon_C1": "epsilon_C1",
                    "epsilon_F1": "epsilon_F_Hx"}

        r32_map_dict = {"sigma_C": "sigma_C1",
                        "sigma_F": "sigma_F_H2",
                        "sigma_H": "sigma_H1",
                        "epsilon_C": "epsilon_C1",
                        "epsilon_F": "epsilon_F_H2",
                        "epsilon_H": "epsilon_H1"}

        r50_map_dict = {"sigma_C1": "sigma_C1",
                        "sigma_H1": "sigma_H1",
                        "epsilon_C1": "epsilon_C1",
                        "epsilon_H1": "epsilon_H1"}

        r125_map_dict = {"sigma_C1":"sigma_C2",
                    "sigma_C2": "sigma_C2",
                    "sigma_F1": "sigma_F_Hx",
                    "sigma_F2":"sigma_F_H1",
                    "sigma_H1": "sigma_H1",
                    "epsilon_C1": "epsilon_C2",
                    "epsilon_C2": "epsilon_C2",
                    "epsilon_F1": "epsilon_F_Hx",
                    "epsilon_F2": "epsilon_F_H1",
                    "epsilon_H1": "epsilon_H1"}

        r134a_map_dict = {"sigma_C1": "sigma_C2",
                    "sigma_C2": "sigma_C2",
                    "sigma_F1": "sigma_F_Hx",
                    "sigma_F2": "sigma_F_H2",
                    "sigma_H1": "sigma_H1",
                    "epsilon_C1": "epsilon_C2",
                    "epsilon_C2": "epsilon_C2",
                    "epsilon_F1": "epsilon_F_Hx",
                    "epsilon_F2": "epsilon_F_H2",
                    "epsilon_H1":"epsilon_H1"}

        r143a_map_dict = {"sigma_C1": "sigma_C2",
                    "sigma_C2": "sigma_C2",
                    "sigma_F1": "sigma_F_Hx",
                    "sigma_H1": "sigma_H1",
                    "epsilon_C1": "epsilon_C2",
                    "epsilon_C2": "epsilon_C2",
                    "epsilon_F1": "epsilon_F_Hx",
                    "epsilon_H1": "epsilon_H1"}

        r170_map_dict = {"sigma_C1": "sigma_C2",
                        "sigma_H1": "sigma_H1",
                        "epsilon_C1": "epsilon_C2",
                        "epsilon_H1": "epsilon_H1"}

        at_names = ["sigma_C1", "sigma_C2", "sigma_H1","sigma_F_H2","sigma_F_H1","sigma_F_Hx",
                "epsilon_C1", "epsilon_C2", "epsilon_H1","epsilon_F_H2","epsilon_F_H1","epsilon_F_Hx"]
        
        molec_map_dicts = {"R14":r14_map_dict,
            "R32":r32_map_dict,
            "R50":r50_map_dict,
            "R125":r125_map_dict,
            "R134a":r134a_map_dict,
            "R143a":r143a_map_dict,
            "R170":r170_map_dict}
        
        super().__init__(at_bounds, at_names, molec_map_dicts)
        #Get scaled bounds
        self.scale_bounds()


class AT_Scheme_9(Atom_Types):
    """
    Class for Atom Type Scheme 9

    Methods
    -------
    __init__(self)
    """
    def __init__(self):
        #Get Bounds
        at_param_bounds_l = [2, 2, 2, 1.5, 2, 2, 2, 10, 10, 10,  2, 15, 15, 15] #Units of Angstroms and Kelvin for Sigmas and Epsilons
        at_param_bounds_u = [4, 4, 4,  3, 4, 4, 4, 75, 75, 75, 10, 50, 50, 50]
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T
        self.scheme_name = "at_9"
        #Get Names
        at_keys = ["sigma_C1", "sigma_C2_F0", "sigma_C2_Fx", "sigma_H1","sigma_F_H2","sigma_F_H1","sigma_F_Hx",
           "epsilon_C1", "epsilon_C2_F0", "epsilon_C2_Fx", "epsilon_H1","epsilon_F_H2","epsilon_F_H1","epsilon_F_Hx"]

        #Create a file that maps param names (keys) to at_param names for atom type 7 (values) for each molecule
        r14_map_dict = {"sigma_C1": "sigma_C1",
                    "sigma_F1": "sigma_F_Hx",
                    "epsilon_C1": "epsilon_C1",
                    "epsilon_F1": "epsilon_F_Hx"}

        r32_map_dict = {"sigma_C": "sigma_C1",
                        "sigma_H": "sigma_H1",
                        "sigma_F": "sigma_F_H2",
                        "epsilon_C": "epsilon_C1",
                        "epsilon_H": "epsilon_H1",
                        "epsilon_F": "epsilon_F_H2"
                        }

        r50_map_dict = {"sigma_C1": "sigma_C1",
                        "sigma_H1": "sigma_H1",
                        "epsilon_C1": "epsilon_C1",
                        "epsilon_H1": "epsilon_H1"}

        r125_map_dict = {"sigma_C1":"sigma_C2_Fx",
                    "sigma_C2": "sigma_C2_Fx",
                    "sigma_F1": "sigma_F_Hx",
                    "sigma_F2":"sigma_F_H1",
                    "sigma_H1": "sigma_H1",
                    "epsilon_C1": "epsilon_C2_Fx",
                    "epsilon_C2": "epsilon_C2_Fx",
                    "epsilon_F1": "epsilon_F_Hx",
                    "epsilon_F2": "epsilon_F_H1",
                    "epsilon_H1": "epsilon_H1"}

        r134a_map_dict = {"sigma_C1": "sigma_C2_Fx",
                    "sigma_C2": "sigma_C2_Fx",
                    "sigma_F1": "sigma_F_Hx",
                    "sigma_F2": "sigma_F_H2",
                    "sigma_H1": "sigma_H1",
                    "epsilon_C1": "epsilon_C2_Fx",
                    "epsilon_C2": "epsilon_C2_Fx",
                    "epsilon_F1": "epsilon_F_Hx",
                    "epsilon_F2": "epsilon_F_H2",
                    "epsilon_H1":"epsilon_H1"}

        r143a_map_dict = {"sigma_C1": "sigma_C2_Fx",
                    "sigma_C2": "sigma_C2_F0",
                    "sigma_F1": "sigma_F_Hx",
                    "sigma_H1": "sigma_H1",
                    "epsilon_C1": "epsilon_C2_Fx",
                    "epsilon_C2": "epsilon_C2_F0",
                    "epsilon_F1": "epsilon_F_Hx",
                    "epsilon_H1": "epsilon_H1"}

        r170_map_dict = {"sigma_C1": "sigma_C2_F0",
                        "sigma_H1": "sigma_H1",
                        "epsilon_C1": "epsilon_C2_F0",
                        "epsilon_H1": "epsilon_H1"}

        at_names = ["sigma_C1", "sigma_C2_F0", "sigma_C2_Fx", "sigma_H1","sigma_F_H2","sigma_F_H1","sigma_F_Hx",
                "epsilon_C1", "epsilon_C2_F0", "epsilon_C2_Fx", "epsilon_H1","epsilon_F_H2","epsilon_F_H1","epsilon_F_Hx"]
        
        molec_map_dicts = {"R14":r14_map_dict,
            "R32":r32_map_dict,
            "R50":r50_map_dict,
            "R125":r125_map_dict,
            "R134a":r134a_map_dict,
            "R143a":r143a_map_dict,
            "R170":r170_map_dict}
        
        super().__init__(at_bounds, at_names, molec_map_dicts)
        #Get scaled bounds
        self.scale_bounds()