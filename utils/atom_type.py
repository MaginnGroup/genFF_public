from .r14 import R14Constants
from .r32 import R32Constants
from .r50 import R50Constants
from .r125 import R125Constants
from .r134a import R134aConstants
from .r143a import R143aConstants
from .r170 import R170Constants
import numpy as np

class Atom_Types:
    """
    Base class for atom typing schemes
    """

    def __init__(self, at_bounds, at_names, molec_map_dicts):
        self.at_bounds = at_bounds
        self.at_names = at_names
        self.molec_map_dicts = molec_map_dicts
        self.at_matrices = {}

    def get_transformation_matrix(self, molec_key):
        if not molec_key in self.at_matrices:
            map_dict = self.molec_map_dicts[molec_key]
            #Create a matrix based on the keys and map dict length
            at_matrix = np.zeros((len(self.at_names), len(map_dict)))

            # Fill at_matrix with ones or zeros based on the presence of keys
            for i, value in enumerate(self.at_names):
                if value in map_dict.values():
                    at_matrix[i, list(map_dict.values()).index(value)] = 1

            self.at_matrices[molec_key] = at_matrix
        else:
            at_matrix = self.at_matrices[molec_key]
        return at_matrix
    
    def check_for_duplicates(self):
        arr_list = list(self.at_matrices.values())
        tuple_list = [tuple(map(tuple, arr)) for arr in arr_list]
        len(tuple_list) != len(set(tuple_list))

        return len(tuple_list) != len(set(tuple_list))
    

class AT_Scheme_7(Atom_Types):
    def __init__(self):
        #Get Bounds
        at_param_bounds_l = [2, 2, 1.5, 2, 2, 2, 10, 10,  2, 15, 15, 15] 
        at_param_bounds_u = [4, 4,   3, 4, 4, 4, 75, 75, 10, 50, 50, 50]
        at_bounds = np.array([at_param_bounds_l, at_param_bounds_u]).T

        #Get Names
        at_keys = ["sigma_C1", "sigma_C2", "sigma_H1","sigma_F_H2","sigma_F_H1","sigma_F_Hx",
           "epsilon_C1", "epsilon_C2", "epsilon_H1","epsilon_F_H2","epsilon_F_H1","epsilon_F_Hx"]

        #Create a file that maps param names to at_param names for atom type 7 for each molecule
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

        r170_map_dict = {"sigma_C2": "sigma_C1",
                        "sigma_H1": "sigma_H1",
                        "epsilon_C2": "epsilon_C1",
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