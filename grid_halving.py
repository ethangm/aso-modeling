import molli as ml
import molli.ncsa_workflow.generate_conformers as gc
from molli.external import _rdkit
from rdkit.Chem import AllChem, rdmolfiles
from pathlib import Path


def BOX_halver_mlib(input: ml.MoleculeLibrary, filepath: str | Path) -> ml.MoleculeLibrary: #FOCUSE ON CLIB
    """
    Will take advantage of the C2 symmetry of BOX molecules to delete half of the molecule
    The "side" does not matter, but it must be consistent between all molecules in library,
    this must be done after BOX alignment
    """
    try:
        new_lib = ml.MoleculeLibrary.new(filepath)
    except Exception as exp:
        print(f"Error with initializing new molecule library: {exp}")
        return
    
    alignment_atoms = gc.get_oxazaline_alignment_atoms(input[0])

    reference_dict = _rdkit.create_rdkit_mol(ref_mol)
    reference_molecule = reference_dict[ref_mol]
    alignment_atoms = list(get_oxazaline_alignment_atoms(reference_molecule))
    
    with new_lib:
        for mol in input:
            new_lib.append(halver(mol, alignment_atoms))

    return new_lib



def BOX_halver_clib(input: ml.ConformerLibrary, filepath: str | Path) -> ml.ConformerLibrary:
    """
    Will take advantage of the C2 symmetry of BOX molecules to delete half of each conformer
    in the library
    """
    try:
        new_lib = ml.ConformerLibrary.new(filepath)
    except Exception as exp:
        print(f"Error with initializing new molecule library: {exp}")
        return
    
#    alignment_atoms = molli_oxazaline_alignment_atoms(input[0][0]) # align on first conformer of first ensemble
    with new_lib:
        for ensemble in input:
            mol2_string = ""
            for conf in ensemble:
                halved = halver(conf)
                if halved is None:
                    break # for 37_1_1_20 issue
                mol2_string += halved.dumps_mol2()
            print(ensemble)
            new_lib.append(ensemble.name, ml.ConformerEnsemble.loads_mol2(mol2_string))

    return new_lib
            

"""
def rd_to_molli():
    temp = rdmolfiles.MolToMolBlock(
                mol, confId=conf[0]
            )  # turn each conformer in mol into temporary molblock string
            ob_temp = pb.readstring("mol", temp)  # read that string into open_babel molecule
            mol2_string += ob_temp.write(
                format="mol2"
            )  # write out ob_temp as mol2 string, append to mol2_string
"""

def halver(mol: ml.Molecule) -> ml.Molecule:
    """
    The workhorse. Only works without attached metal (e.g. no CuCl2).
    From BOX substructure, deletes first C2 carbon. Then takes first N atom and
    deletes all connected atoms (now disconnected from methylene bridge and other half)
    """
    reference_atoms = molli_oxazaline_alignment_atoms(mol) #finds alignment atoms for each mol

    if reference_atoms is None: # for 37_1_1_20 issue
        return None
    
    first_N = reference_atoms[0]
    first_C2 = reference_atoms[1]

    new_mol = ml.Molecule(mol)

    new_mol.del_atom(first_C2)

    if first_C2 < first_N:
        first_N -= 1    # if index of first N should be changed!

    to_delete = dfs(first_N, new_mol)
    to_delete = sorted(to_delete, reverse=True)
    for atom in to_delete:
        new_mol.del_atom(atom)

    return new_mol
    

def dfs(start, mol) -> list:
    """
    Returns all indices of atoms in mol found during search
    """
    stack = [start]
    visited = []

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.append(current)
            stack.extend(mol.get_atom_index(i) for i in mol.connected_atoms(current))

    return visited


def molli_oxazaline_alignment_atoms(mol) -> list:
    """
    Reimplementation of 'get_oxazaline_alignment_atoms' algorithm for usage with molli Molecules
    instead of rdkit molecules

    argument mol: molli molecule

    returns: list of atom indices of the nitrogen in one ring, C2 of that ring,
    the bridging methylene, the C2 of the other ring, and the nitrogen of the other ring, in that order.
    """

    # find all the nitrogens, we will use different cases of these to identify the structure
    nitrogens = [i for i in mol.yield_atoms_by_element('N')]
    # print(len(nitrogens))
    first_N, second_N, c2, bridge, other_c2 = None, None, None, None, None
    oxazoline_Ns = []

    if len(nitrogens) == 2:
        oxazoline_Ns = [i for i in mol.yield_atoms_by_element('N')]

    # the case when the bridging group is nitrile
    elif len(nitrogens) == 3:
        # print('here')
        # the nitrile N only has one neighbor
        for atom in nitrogens:
            if len([i for i in mol.connected_atoms(atom)]) != 1:
                oxazoline_Ns.append(atom)

    # the case where we have nitrogens in other parts of the molecule, e.g. pyridine rings
    elif len(nitrogens) == 4:
        oxazoline_Ns = []
        # iterate through nitrogens
        for atom in nitrogens:
            # print('here1')
            neighbors = [i for i in mol.connected_atoms(atom)]
            # iterate through nitrogen neighbors
            for neighbor in neighbors:
                # oxazoline C2 should be SP2
                if neighbor.atype == 32:   # this is sp2
                    # oxazoline C2 should have exactly 3 neighbors, C bridge, N, and O
                    if sorted([i.element.symbol for i in mol.connected_atoms(neighbor)]) == sorted(
                        ["C", "N", "O"]     # that .symbol fixed everything
                    ):
                        # print('here2')
                        oxazoline_Ns.append(atom)
                        break

    # the cases where we have oxazoline nitrogens, nitrogens at 4-position, and a nitrile bridging group, and the case
    # with oxazoline nitrogens, nitrogens at 4-position, and bridging group 29 with 2 nitrogens can be treated similarly (5 and 6 total nitrogens)
    elif (len(nitrogens) == 5) or (len(nitrogens) == 6):
        oxazoline_Ns = []
        # iterate through nitrogens
        for atom in nitrogens:
            neighbors = [i for i in mol.connected_atoms(atom)]
            # this is the nitrile, skip it
            if len(neighbors) == 1:
                continue
            # iterate through nitrogen neighbors
            for neighbor in neighbors:
                # oxazoline C2 should be SP2
                if neighbor.atype == 32:
                    # oxazoline C2 should have exactly 3 neighbors, C bridge, N, and O
                    if sorted([i.element.symbol for i in mol.connected_atoms(neighbor)]) == sorted(
                        ["C", "N", "O"]
                    ):
                        # print('here2')
                        oxazoline_Ns.append(atom)
                        break

    else:
        raise Exception(f"{mol.name} failed! nitrogen count: {len(nitrogens)}")

    try:
        assert len(oxazoline_Ns) == 2
    except AssertionError:
        print(f"incorrect # of nitrogens! {len(oxazoline_Ns)}")
        print(oxazoline_Ns)
        print(mol)
        return
    first_N, second_N = oxazoline_Ns
    # get c2 next to first N

    # this logic still works for the cbr2 linkers (which are CSp3)
    c2 = [
        atom
        for atom in mol.connected_atoms(first_N)
        if atom.atype == 32
    ][0]
    other_c2 = [
        atom
        for atom in mol.connected_atoms(second_N)
        if atom.atype == 32
    ][0]

    # get bridge
    bridge = set([atom for atom in mol.connected_atoms(c2)]).intersection(
        set([atom for atom in mol.connected_atoms(other_c2)])
    )
    assert len(bridge) == 1
    bridge = bridge.pop()

    assert second_N in [i for i in mol.connected_atoms(other_c2)]

    return (
        mol.get_atom_index(first_N), 
        mol.get_atom_index(c2), 
        mol.get_atom_index(bridge), 
        mol.get_atom_index(other_c2), 
        mol.get_atom_index(second_N)
        )


if __name__ == "__main__":
    lib = ml.ConformerLibrary("../nbo-cu-box-clean/caseys/conformers_no_linker.mlib")

    new_lib = BOX_halver_clib(lib, "../nbo-cu-box-clean/halve_testing/all_halved_fixed_2.mlib")

    print("Done!")