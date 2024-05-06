# Ryan DeFever
# 2021 Jul 16
# Maginn Research Group
# University of Notre Dame

# This script uses GMSO; to install gmso into a
# new conda environment
#     conda create --name gmso gmso -c conda-forge
#     conda activate gmso

# This script should not be trusted; at least do
# some sanity checks on the number of bonds/angles/dihedrals
# etc that you get. The best option is to do an
# energy comparison between LAMMPS and GROMACS for a
# molecule or two.


import sys
import datetime
import warnings
import unyt as u
import numpy as np

import sympy

from gmso import Topology, ForceField
from gmso.core.atom import Atom
from gmso.core.bond import Bond
from gmso.core.angle import Angle
from gmso.core.dihedral import Dihedral
from gmso.core.improper import Improper

from gmso.core.atom_type import AtomType
from gmso.core.bond_type import BondType
from gmso.core.angle_type import AngleType
from gmso.core.dihedral_type import DihedralType
from gmso.core.improper_type import ImproperType

from gmso.core.box import Box
from gmso.core.element import element_by_symbol

from gmso.formats.mcf import write_mcf


ALL_SECTION_KEYWORDS = ["SETTING", "ATOM", "BOND", "ANGLE", "DIHEDRAL", "IMPROPER"]


def main():

    if len(sys.argv) != 4:
        print("Usage: python maffa_to_gmx.py [maffa_path] [resname] [outname]")
        exit(1)
    else:
        maffa_path = str(sys.argv[1])
        resname = str(sys.argv[2])
        outname = str(sys.argv[3])

    maffa2gmx(maffa_path, resname, outname)


def maffa2gmx(maffa_path, resname, outname):

    maffa_ff_data, maffa_mol_data = _read_maffa(maffa_path)

    # Extract important idx numbers
    setting_idx_s, setting_idx_e = _find_line_idx(maffa_ff_data, "SETTING")
    atomtype_idx_s, atomtype_idx_e = _find_line_idx(maffa_ff_data, "ATOM")
    bondtype_idx_s, bondtype_idx_e = _find_line_idx(maffa_ff_data, "BOND")
    angletype_idx_s, angletype_idx_e = _find_line_idx(maffa_ff_data, "ANGLE")
    dihedraltype_idx_s, dihedraltype_idx_e = _find_line_idx(maffa_ff_data, "DIHEDRAL")
    impropertype_idx_s, impropertype_idx_e = _find_line_idx(maffa_ff_data, "IMPROPER")

    atom_idx_s, atom_idx_e = _find_line_idx(maffa_mol_data, "ATOM")
    bond_idx_s, bond_idx_e = _find_line_idx(maffa_mol_data, "BOND")
    angle_idx_s, angle_idx_e  = _find_line_idx(maffa_mol_data, "ANGLE")
    dihedral_idx_s, dihedral_idx_e = _find_line_idx(maffa_mol_data, "DIHEDRAL")
    improper_idx_s, improper_idx_e = _find_line_idx(maffa_mol_data, "IMPROPER")

    # Extract the FF information from MaFFA and save to a Forcefield
    ff = ForceField()
    atomtype_dict, element_dict = _read_atomtypes(maffa_ff_data, atomtype_idx_s, atomtype_idx_e)
    ff.atom_types = atomtype_dict
    if bondtype_idx_s is not None:
        bondtype_dict = _read_bondtypes(maffa_ff_data, bondtype_idx_s, bondtype_idx_e)
        ff.bond_types = bondtype_dict
    if angletype_idx_s is not None:
        angletype_dict = _read_angletypes(maffa_ff_data, angletype_idx_s, angletype_idx_e)
        ff.angle_types = angletype_dict
    if dihedraltype_idx_s is not None:
        dihedraltype_dict = _read_dihedraltypes(maffa_ff_data, dihedraltype_idx_s, dihedraltype_idx_e)
        ff.dihedral_types = dihedraltype_dict
    if impropertype_idx_s is not None:
        impropertype_dict = _read_impropertypes(maffa_ff_data, impropertype_idx_s, impropertype_idx_e)
        ff.improper_types = impropertype_dict

    # We have to make a topology so we have sites to add to connections
    top = Topology()
    top.box = Box([3., 3., 3.] * u.nm)
    top.name = resname

    # Create atom/bond/angle/dihedral lists
    atom_list, name_dict = _create_atom_list(top, ff, maffa_mol_data, atom_idx_s, atom_idx_e, element_dict)
    print(f"Atom list length: {len(atom_list)}")
    if bond_idx_s is not None:
        bond_list = _create_bond_list(top, ff, maffa_mol_data, bond_idx_s, bond_idx_e, name_dict)
        print(f"Bond list length: {len(bond_list)}")
    if angle_idx_s is not None:
        angle_list = _create_angle_list(top, ff, maffa_mol_data, angle_idx_s, angle_idx_e, name_dict)
        print(f"Angle list length: {len(angle_list)}")
    if dihedral_idx_s is not None:
        dihedral_list = _create_dihedral_list(top, ff, maffa_mol_data, dihedral_idx_s, dihedral_idx_e, name_dict)
        print(f"Dihedral list length: {len(dihedral_list)}")
    if improper_idx_s is not None:
        improper_list = _create_improper_list(top, ff, maffa_mol_data, improper_idx_s, improper_idx_e, name_dict)
        print(f"Improper list length: {len(improper_list)}")
    else:
        improper_list = []

    # Write GMX files
    with open(outname + ".itp", "w") as f:
        _write_gmx_defaults(f)
        _write_gmx_atomtypes(f, ff, element_dict)
        _write_gmx_moleculetype(f, resname)
        _write_gmx_atoms(f, top, atom_list)
        if dihedral_idx_s is not None:
            _write_gmx_pairs(f, top, bond_list, angle_list, dihedral_list)
        if bond_idx_s is not None:
            _write_gmx_bonds(f, top, bond_list)
        if angle_idx_s is not None:
            _write_gmx_angles(f, top, angle_list)
        if dihedral_idx_s is not None:
            _write_gmx_dihedrals(f, top, dihedral_list, improper_list)

    with open(outname + ".gro", "w") as f:
        _write_gmx_coordinates(f, top, atom_list)


def _find_line_idx(data, string):
    start_idx = None
    end_idx = None
    for idx, line in enumerate(data):
        if start_idx is not None:
            if line[0] in ALL_SECTION_KEYWORDS:
                end_idx = idx
                break
        if line[0] == string:
            start_idx = idx+1

    if start_idx is None:
        warnings.warn(f"string: {string} not found")
    if end_idx is None:
        end_idx = len(data)

    return start_idx, end_idx


def _read_maffa(filen):

    maffa_data = []
    with open(filen) as f:
        for line in f:
            if line[0][0] == "!" or line[0][0] == "#":
                pass
            elif len(line.split()) == 0:
                pass
            else:
                maffa_data.append(line.split())

    # Verify format
    #if maffa_data[0][1] != "M02":
    #    raise ValueError("Invalid MAFFA format")

    # Split MaFFA into two section, FF definition and molecule definition
    for idx, line in enumerate(maffa_data):
        if line[0] == "MOLECULE":
            molecule_idx = idx

    return maffa_data[:molecule_idx], maffa_data[molecule_idx:]

def _read_atomtypes(maffa_data, atomtype_idx_s, atomtype_idx_e):

    ## Atomtypes
    atomtype_dict = {}
    element_dict = {}
    for idx in range(atomtype_idx_s, atomtype_idx_e):
        (type_, mass, element, epsilon, sigma, *_) = maffa_data[idx]
        atomtype_dict[type_] = AtomType(
            name=type_,
            mass=float(mass) * u.amu,
            expression="4*epsilon*((sigma/r)**12 - (sigma/r)**6)",
            independent_variables="r",
            parameters={
                "sigma": float(sigma) * u.angstrom,
                "epsilon": float(epsilon) * u.K * u.kb
            },
        )
        element_dict[type_] = element

    return atomtype_dict, element_dict


def _read_bondtypes(maffa_data, bondtype_idx_s, bondtype_idx_e):

    ## Bondtypes
    bondtype_dict = {}
    for idx in range(bondtype_idx_s, bondtype_idx_e):
        (type1, type2, style, k, r_eq, *_) = maffa_data[idx]
        if style != "harmonic":
            raise ValueError("Unsupported bond type!")
        bondtype_dict[(type1, type2)] = BondType(
            name=f"{type1}~{type2}",
            expression="k * (r-r_eq)**2",
            independent_variables="r",
            parameters={
                "k": float(k) * u.K * u.kb / u.angstrom**2,
                "r_eq": float(r_eq) * u.angstrom,
            }
        )


    return bondtype_dict


def _read_angletypes(maffa_data, angletype_idx_s, angletype_idx_e):
    ## Angletypes
    angletype_dict = {}
    for idx in range(angletype_idx_s, angletype_idx_e):
        (type1, type2, type3, style, k, theta_eq, *_) = maffa_data[idx]
        if style != "harmonic":
            raise ValueError("Unsupported angle type!")
        angletype_dict[(type1, type2, type3)] = AngleType(
            name=f"{type1}~{type2}~{type3}",
            expression="k * (theta-theta_eq)**2",
            independent_variables="theta",
            parameters={
                "k": float(k) * u.K * u.kb / u.rad**2,
                "theta_eq": float(theta_eq) * u.degree,
            }
        )

    return angletype_dict


def _read_dihedraltypes(maffa_data, dihedraltype_idx_s, dihedraltype_idx_e):
    ## Dihedraltypes
    dihedraltype_dict = {}
    for idx in range(dihedraltype_idx_s, dihedraltype_idx_e):
        (type1, type2, type3, type4, style, *param_list) = maffa_data[idx]
        if style == "charmm":
            expr = "kn * (1 + cos(n * a - a0))"
            params = {
                    "kn": float(param_list[0]) * u.K * u.kb,
                    "n": int(param_list[1]) * u.dimensionless,
                    "a0": float(param_list[2]) * u.degree,
            }
        elif style == "opls":
            expr = "(k1*(1+cos(a)) + k2*(1-cos(2*a)) + k3*(1+cos(3*a)) + k4*(1-cos(4*a)))*(1/2)"
            params = {"k"+str(i+1): float(param_list[i]) * u.K * u.kb for i in range(4)}
        else:
            raise ValueError("Unsupported dihedral type: "+style)
        # Need to handle layered dihedrals; list of matching dihedrals
        if (type1, type2, type3, type4) not in dihedraltype_dict:
            dihedraltype_dict[(type1, type2, type3, type4)] = [
                DihedralType(
                    name=f"{type1}~{type2}~{type3}~{type4}",
                    expression=expr,
                    independent_variables="a",
                    parameters=params
                )
            ]

        else:
            dihedraltype_dict[(type1, type2, type3, type4)].append(
                DihedralType(
                    name=f"{type1}~{type2}~{type3}~{type4}",
                    expression=expr,
                    independent_variables="a",
                    parameters=params
                )
            )

    return dihedraltype_dict


def _read_impropertypes(maffa_data, impropertype_idx_s, impropertype_idx_e):
    # Impropertypes
    impropertype_dict = {}
    for idx in range(impropertype_idx_s, impropertype_idx_e):
        (type1, type2, type3, type4, style, kn, n, a0, *_) = maffa_data[idx]
        if style != "cosine":
            raise ValueError("Unsupported dihedral type!")
        # Handle layered impropers
        if (type1, type2, type3, type4) not in impropertype_dict:
            impropertype_dict[(type1, type2, type3, type4)] = [
                ImproperType(
                    name=f"{type1}~{type2}~{type3}~{type4}",
                    expression="kn * (1 + cos(n * a - a0))",
                    independent_variables="a",
                    parameters={
                        "kn": float(kn) * u.K * u.kb,
                        "n": int(n) * u.dimensionless,
                        "a0": float(a0) * u.degree,

                    }
                )
            ]
        else:
            impropertype_dict[(type1, type2, type3, type4)].append(
                ImproperType(
                    name=f"{type1}~{type2}~{type3}~{type4}",
                    expression="kn * (1 + cos(n * a - a0))",
                    independent_variables="a",
                    parameters={
                        "kn": float(kn) * u.K * u.kb,
                        "n": int(n) * u.dimensionless,
                        "a0": float(a0) * u.degree,

                    }
                )
            )

    return impropertype_dict


def _create_atom_list(top, ff, maffa_data, atom_idx_s, atom_idx_e, element_dict):
    # Atoms
    atom_list = []
    atomtype_dict = ff.atom_types
    name_dict = {}
    for idx in range(atom_idx_s, atom_idx_e):
        (name, type_, charge, x, y, z, *_) = maffa_data[idx]
        atom =  Atom(
            name=name,
            atom_type=atomtype_dict[type_],
            charge=float(charge) * u.elementary_charge,
            position = np.array([x, y, z], dtype=np.float64) * u.angstrom,
            element=element_by_symbol(element_dict[type_]),
        )

        top.add_site(atom)
        atom_list.append(atom)
        name_dict[name] = idx - atom_idx_s

    top.update_topology()
    return atom_list, name_dict


def _create_bond_list(top, ff, maffa_data, bond_idx_s, bond_idx_e, name_dict):
    # Bonds
    bond_list = []
    bondtype_dict = ff.bond_types
    for idx in range(bond_idx_s, bond_idx_e):
        (name1, name2, *_) = maffa_data[idx]
        site1 = top.sites[name_dict[name1]]
        site2 = top.sites[name_dict[name2]]
        type1 = site1.atom_type.name
        type2 = site2.atom_type.name
        bond_type = None
        for (btype1, btype2) in bondtype_dict.keys():
            if type1 == btype1 and type2 == btype2:
                bond_type = bondtype_dict[(btype1, btype2)]
            elif type2 == btype1 and type1 == btype2:
                bond_type = bondtype_dict[(btype1, btype2)]
        if bond_type is None:
            raise ValueError("No bond type found for atom types "+type1+" and "+type2)
        bond_list.append(
            Bond(
                name=f"{name1}~{name2}",
                bond_type=bond_type,
                connection_members=[
                    site1,
                    site2,
                ],
            )
        )

    return bond_list

def _create_angle_list(top, ff, maffa_data, angle_idx_s, angle_idx_e, name_dict):
    # Angles
    angle_list = []
    angletype_dict = ff.angle_types
    for idx in range(angle_idx_s, angle_idx_e):
        (name1, name2, name3, *_) = maffa_data[idx]
        site1 = top.sites[name_dict[name1]]
        site2 = top.sites[name_dict[name2]]
        site3 = top.sites[name_dict[name3]]
        type1 = site1.atom_type.name
        type2 = site2.atom_type.name
        type3 = site3.atom_type.name

        angle_type = None
        for (atype1, atype2, atype3) in angletype_dict.keys():
            if type2 == atype2 and type1 == atype1 and type3 == atype3:
                angle_type = angletype_dict[(atype1, atype2, atype3)]
            if type2 == atype2 and type3 == atype1 and type1 == atype3:
                angle_type = angletype_dict[(atype1, atype2, atype3)]
        if angle_type is None:
            raise ValueError("No angle type found!")
        angle_list.append(
            Angle(
                name=f"{name1}~{name2}~{name3}",
                angle_type=angle_type,
                connection_members=[
                    site1,
                    site2,
                    site3,
                ],
            )
        )

    return angle_list


def _create_dihedral_list(top, ff, maffa_data, dihedral_idx_s, dihedral_idx_e, name_dict):
    # Dihedrals
    dihedral_list = []
    dihedraltype_dict = ff.dihedral_types
    for idx in range(dihedral_idx_s, dihedral_idx_e):
        (name1, name2, name3, name4, *_) = maffa_data[idx]
        site1 = top.sites[name_dict[name1]]
        site2 = top.sites[name_dict[name2]]
        site3 = top.sites[name_dict[name3]]
        site4 = top.sites[name_dict[name4]]

        type1 = site1.atom_type.name
        type2 = site2.atom_type.name
        type3 = site3.atom_type.name
        type4 = site4.atom_type.name

        dihedral_types = None
        for (dtype1, dtype2, dtype3, dtype4) in dihedraltype_dict.keys():
            if type1 == dtype1 and type2 == dtype2 and type3 == dtype3 and type4 == dtype4:
                dihedral_types = dihedraltype_dict[(dtype1, dtype2, dtype3, dtype4)]
            if type4 == dtype1 and type3 == dtype2 and type2 == dtype3 and type1 == dtype4:
                dihedral_types = dihedraltype_dict[(dtype1, dtype2, dtype3, dtype4)]
        if dihedral_types is None:
            raise ValueError(f"No dihedral type found! {type1} {type2} {type3} {type4}")
        for dihedral_type in dihedral_types:
            dihedral_list.append(
                Dihedral(
                    name=f"{name1}~{name2}~{name3}~{name4}",
                    dihedral_type=dihedral_type,
                    connection_members=[
                        site1,
                        site2,
                        site3,
                        site4,
                    ],
                )
            )

    return dihedral_list


def _create_improper_list(top, ff, maffa_data, improper_idx_s, improper_idx_e, name_dict):
    # Impropers (treated as dihedrals -- since amber-style)
    improper_list = []
    impropertype_dict = ff.improper_types
    for idx in range(improper_idx_s, improper_idx_e):
        (name1, name2, name3, name4, *_) = maffa_data[idx]
        site1 = top.sites[name_dict[name1]]
        site2 = top.sites[name_dict[name2]]
        site3 = top.sites[name_dict[name3]]
        site4 = top.sites[name_dict[name4]]

        type1 = site1.atom_type.name
        type2 = site2.atom_type.name
        type3 = site3.atom_type.name
        type4 = site4.atom_type.name

        improper_types = None
        for (itype1, itype2, itype3, itype4) in impropertype_dict.keys():
            if type1 == itype1 and type2 == itype2 and type3 == itype3 and type4 == itype4:
                improper_types = impropertype_dict[(itype1, itype2, itype3, itype4)]
            if type4 == itype1 and type3 == itype2 and type2 == itype3 and type1 == itype4:
                improper_types = impropertype_dict[(itype1, itype2, itype3, itype4)]

        if improper_types is None:
            raise ValueError("No improper type found!")
        for improper_type in improper_types:
            improper_list.append(
                Improper(
                    name=f"{name1}~{name2}~{name3}~{name4}",
                    improper_type=improper_type,
                    connection_members=[
                        site1,
                        site2,
                        site3,
                        site4,
                    ],
                )
            )

    return improper_list


def _write_gmx_defaults(f):
    """DEFAULTS HARD CODED FOR GAFF"""
    f.write("; GROMACS topology file\n")
    f.write(f"; Written by maffa_to_gmx.py on {datetime.datetime.now()}\n")
    f.write("\n\n")
    f.write("[ defaults ]\n")
    f.write("; nbfunc    comb_rule    gen_pairs     fudge_LJ     fudge_QQ\n")
    f.write("  1         2            yes           0.5          0.8333\n")
    f.write("\n")
    f.write("\n")


def _write_gmx_atomtypes(f, ff, element_dict):
    atomtype_dict = ff.atom_types
    f.write("[ atomtypes ]\n")
    f.write("; atype       at.num     mass    charge   ptype   sigma    epsilon\n")
    for type_, atype in atomtype_dict.items():
        f.write(
            "  {name:12s}{atnum:5d}{mass:10.4f}{charge:8.4f}{ptype:>5s}{sigma:12.6f}{epsilon:12.6f}\n".format(
                name=type_,
                atnum=element_by_symbol(element_dict[type_]).atomic_number,
                mass=atype.mass.in_units(u.amu).value,
                charge=0.0,
                ptype="A",
                sigma=atype.parameters["sigma"].in_units("nm").value,
                epsilon=atype.parameters["epsilon"].in_units("kJ/mol").value,
            )
        )
    f.write("\n")
    f.write("\n")


def _write_gmx_moleculetype(f, molname):
    """DEFAULTS HARD CODED FOR GAFF"""
    f.write("[ moleculetype ]\n")
    f.write("; molname    nrexcl\n")
    f.write(f"{molname:12s} 3\n")
    f.write("\n")
    f.write("\n")


def _write_gmx_atoms(f, top, atom_list):
    f.write("[ atoms ]\n")
    f.write("; atnr       atype   resnr resnm atname chgrp    charge    mass\n")
    for atom in atom_list:
        f.write(
            "  {atnr:5d}{atype:>12s}{resnr:5d}{resnm:>6s}{atname:>7s}{atnr:5d}{charge:15.8f}{mass:12.4f}\n".format(
                atnr=top.get_index(atom)+1,
                atype=atom.atom_type.name,
                resnr=1,
                resnm=top.name,
                atname=atom.name,
                charge=atom.charge.in_units(u.elementary_charge).value,
                mass=atom.atom_type.mass.in_units(u.amu).value,
            )
        )
    f.write("\n")
    f.write("\n")


def _write_gmx_pairs(f, top, bond_list, angle_list, dihedral_list):
    pairs_list = []
    for dihedral in dihedral_list:
        idx1 = top.get_index(dihedral.connection_members[0])+1
        idx4 = top.get_index(dihedral.connection_members[3])+1
        if idx1 < idx4:
            pairs_list.append((idx1, idx4))
        else:
            pairs_list.append((idx4, idx1))

    # Only want unique cases
    pairs_set = set(pairs_list)

    # Now filter out any bonds/angles
    # These are by def 1-2 and 1-3 interactions
    # In a ring they could also be a 1-4 interaction (other way around)
    ring_pairs = []
    for (pidx1, pidx2) in pairs_set:
        for bond in bond_list:
            bidx1 = top.get_index(bond.connection_members[0])+1
            bidx2 = top.get_index(bond.connection_members[1])+1
            if pidx1 == bidx1 and pidx2 == bidx2:
                ring_pairs.append((pidx1, pidx2))
            elif pidx1 == bidx2 and pidx2 == bidx1:
                ring_pairs.append((pidx1, pidx2))
        for angle in angle_list:
            aidx1 = top.get_index(angle.connection_members[0])+1
            aidx2 = top.get_index(angle.connection_members[2])+1
            if pidx1 == aidx1 and pidx2 == aidx2:
                ring_pairs.append((pidx1, pidx2))
            elif pidx1 == aidx2 and pidx2 == aidx1:
                ring_pairs.append((pidx1, pidx2))

    for pair in ring_pairs:
        pairs_set.remove(pair)

    f.write("[ pairs ]\n")
    f.write("; idx1   idx2  funct\n")
    for (idx1, idx4) in pairs_set:
        f.write(
            "  {idx1:6d}{idx4:>6d}{funct:5d}\n".format(
                idx1=idx1,
                idx4=idx4,
                funct=1,
            )
        )
    f.write("\n")
    f.write("\n")



def _write_gmx_bonds(f, top, bond_list):
    """MAFFA TO GMX FACTOR OF 2 CONVERSION HARD-CODED"""
    f.write("[ bonds ]\n")
    f.write("; idx1   idx2   funct   b0     kb\n")
    for bond in bond_list:
        f.write(
            "  {idx1:6d}{idx2:>6d}{funct:5d}{b0:12.5f}{kb:15.3f}\n".format(
                idx1=top.get_index(bond.connection_members[0])+1,
                idx2=top.get_index(bond.connection_members[1])+1,
                funct=1,
                b0=bond.connection_type.parameters["r_eq"].in_units(u.nanometer).value,
                kb=2*bond.connection_type.parameters["k"].in_units(u.kJ/u.mol/u.nanometer**2).value,
            )
        )
    f.write("\n")
    f.write("\n")


def _write_gmx_angles(f, top, angle_list):
    """MAFFA TO GMX FACTOR OF 2 CONVERSION HARD-CODED"""
    f.write("[ angles ]\n")
    f.write("; idx1   idx2    idx3  funct   theta0     k_theta\n")
    for angle in angle_list:
        f.write(
            "  {idx1:6d}{idx2:>6d}{idx3:>6d}{funct:5d}{t0:12.3f}{kt:15.3f}\n".format(
                idx1=top.get_index(angle.connection_members[0])+1,
                idx2=top.get_index(angle.connection_members[1])+1,
                idx3=top.get_index(angle.connection_members[2])+1,
                funct=1,
                t0=angle.connection_type.parameters["theta_eq"].in_units(u.degree).value,
                kt=2*angle.connection_type.parameters["k"].in_units(u.kJ/u.mol/u.radian**2).value,
            )
        )
    f.write("\n")
    f.write("\n")


def _write_gmx_dihedrals(f, top, dihedral_list, improper_list):
    opls_to_rb_mat = np.zeros((4,6))
    opls_to_rb_mat[1,0] = 1
    opls_to_rb_mat[[0,2],0] = 0.5
    opls_to_rb_mat[0,1] = -0.5
    opls_to_rb_mat[2,1] = 1.5
    opls_to_rb_mat[1,2] = -1
    opls_to_rb_mat[3,2] = 4
    opls_to_rb_mat[2,3] = -2
    opls_to_rb_mat[3,4] = -4
    f.write("[ dihedrals ]\n")
    charmm_expr = sympy.sympify("kn * (1 + cos(n * a - a0))")
    opls_expr = sympy.sympify("(k1*(1+cos(a)) + k2*(1-cos(2*a)) + k3*(1+cos(3*a)) + k4*(1-cos(4*a)))*(1/2)")
    charmm_dihedral_list = [dihedral for dihedral in dihedral_list if dihedral.connection_type.expression == charmm_expr]
    opls_dihedral_list = [dihedral for dihedral in dihedral_list if dihedral.connection_type.expression == opls_expr]
    if charmm_dihedral_list:
        f.write("; idx1   idx2    idx3    idx4  funct  theta_s     kn     mult\n")
    for dihedral in charmm_dihedral_list:
        f.write(
            "  {idx1:6d}{idx2:>6d}{idx3:>6d}{idx4:>6d}{funct:5d}{ts:12.3f}{kt:15.3f}{mult:5d}\n".format(
                idx1=top.get_index(dihedral.connection_members[0])+1,
                idx2=top.get_index(dihedral.connection_members[1])+1,
                idx3=top.get_index(dihedral.connection_members[2])+1,
                idx4=top.get_index(dihedral.connection_members[3])+1,
                funct=1,
                ts=dihedral.connection_type.parameters["a0"].in_units(u.degree).value,
                kt=dihedral.connection_type.parameters["kn"].in_units(u.kJ/u.mol).value,
                mult=dihedral.connection_type.parameters["n"].value,
            )
        )
    if opls_dihedral_list:
        f.write("; RB dihedrals\n")
        f.write("; idx1   idx2    idx3    idx4  funct  C0   C1   C2   C3   C4   C5\n")
    for dihedral in opls_dihedral_list:
        opls_param_dict = dihedral.connection_type.parameters
        opls_params = np.array([opls_param_dict["k"+str(i+1)].to_value("kJ/mol") for i in range(4)])
        rb_params = opls_params @ opls_to_rb_mat
        f.write(
            "  {idx1:6d}{idx2:>6d}{idx3:>6d}{idx4:>6d}{funct:5d}{C0:15.3f}{C1:15.3f}{C2:15.3f}{C3:15.3f}{C4:15.3f}{C5:15.3f}\n".format(
                idx1=top.get_index(dihedral.connection_members[0])+1,
                idx2=top.get_index(dihedral.connection_members[1])+1,
                idx3=top.get_index(dihedral.connection_members[2])+1,
                idx4=top.get_index(dihedral.connection_members[3])+1,
                funct=3,
                C0 = rb_params[0],
                C1 = rb_params[1],
                C2 = rb_params[2],
                C3 = rb_params[3],
                C4 = rb_params[4],
                C5 = rb_params[5],
            )
        )
    f.write("; Impropers\n")
    f.write("; idx1   idx2    idx3    idx4  funct  theta_s     kn     mult\n")
    for improper in improper_list:
        f.write(
            "  {idx1:6d}{idx2:>6d}{idx3:>6d}{idx4:>6d}{funct:5d}{ts:12.3f}{kt:15.3f}{mult:5d}\n".format(
                idx1=top.get_index(improper.connection_members[0])+1,
                idx2=top.get_index(improper.connection_members[1])+1,
                idx3=top.get_index(improper.connection_members[2])+1,
                idx4=top.get_index(improper.connection_members[3])+1,
                funct=4,
                ts=improper.connection_type.parameters["a0"].in_units(u.degree).value,
                kt=improper.connection_type.parameters["kn"].in_units(u.kJ/u.mol).value,
                mult=improper.connection_type.parameters["n"].value,
            )
        )
    f.write("\n")
    f.write("\n")


def _write_gmx_coordinates(f, top, atom_list):
    """Write coordinates to .gro file and center in box"""
    # Calculate cog
    cog = np.array([0.0, 0.0, 0.0])
    for atom in atom_list:
        cog += np.array([
            atom.position[0].in_units(u.nanometer).value,
            atom.position[1].in_units(u.nanometer).value,
            atom.position[2].in_units(u.nanometer).value,
        ])
    cog /= len(top.sites)

    boxx = top.box.lengths[0].in_units(u.nanometer).value
    boyy = top.box.lengths[1].in_units(u.nanometer).value
    bozz = top.box.lengths[2].in_units(u.nanometer).value

    f.write(f"Written by maffa_to_gmx.py on {datetime.datetime.now()}\n")
    f.write(f"{len(top.sites):5d}\n")
    for atom in atom_list:
        f.write(
            "{resnr:5d}{resnm:<5s}{atname:5s}{atnr:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n".format(
                resnr=1,
                resnm=top.name,
                atname=atom.name,
                atnr=top.get_index(atom)+1,
                x=atom.position[0].in_units(u.nanometer).value-cog[0]+boxx/2.0,
                y=atom.position[1].in_units(u.nanometer).value-cog[1]+boyy/2.0,
                z=atom.position[2].in_units(u.nanometer).value-cog[2]+bozz/2.0,
            )
        )
    f.write("{boxx:10.5f}{boyy:10.5f}{bozz:10.5f}\n".format(
            boxx=boxx,
            boyy=boyy,
            bozz=bozz,
        )
    )


if __name__ == "__main__":
    main()

