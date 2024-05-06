#Imports
from utils.molec_class_files import r14, r32, r50, r125, r134a, r143a, r170
from utils import atom_type, opt_atom_types
import numpy as np
import unyt as u
import pandas as pd
import os
import copy
import scipy 

#After jobs are finished
#save signac results for each atom for a given atom typing scheme and number of training parameters
#Plot Pvap and Hvap vs T and compare to GAFF, exp, our old results, and literature