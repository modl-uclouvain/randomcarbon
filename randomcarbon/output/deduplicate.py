import itertools
import multiprocessing
from functools import partial
from typing import List, Dict
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from randomcarbon.output.store import MongoStore


def deduplicate_list(structures: List[Structure], stol: float = 0.03, primitive: bool = True) -> List[int]:

    # if required, do the preprocessing for all the structures before. no scaling.
    if primitive:
        for i, s in enumerate(structures):
            spga = SpacegroupAnalyzer(s)
            structures[i] = spga.find_primitive()

    sm = StructureMatcher(primitive_cell=False, stol=stol, scale=False)

    duplicated_data = []
    unique_structures = {}
    for i_test, s_test in enumerate(structures):

        for i_ref, s_ref in unique_structures.items():
            # if len(s_test) != len(s_ref):
            #     continue
            # access private method since this will save some time in avoiding the
            # preprocessing.
            match = sm._match(s_test, s_ref, 1, True, break_on_match=True)
            if match is not None and match[0] <= sm.stol:
                duplicated_data.append(i_ref)
                break
        else:
            unique_structures[i_test] = s_test
            duplicated_data.append(None)

    return duplicated_data


def group_data(data: List[Dict], energy_tol: float = 0.03) -> List[List[Dict]]:
    sorted_data = sorted(data, key=lambda d: (d["nsites"], d["n_inequivalent_sites"], d["energy_per_atom"]))
    key_func = lambda d: (d["nsites"], d["n_inequivalent_sites"], (d["energy_per_atom"] // energy_tol) * energy_tol)
    grouped_data = []
    for _, group in itertools.groupby(sorted_data, key=key_func):
        grouped_data.append(list(group))

    return grouped_data


def run_deduplicate(data: List[Dict], connection_data: dict, stol: float = 0.03, primitive: bool = True,
                    delete: bool = False):

    duplicated = deduplicate_list([Structure.from_dict(d["structure"]) for d in data], stol=stol, primitive=primitive)

    mongo_store = MongoStore(**connection_data)
    mongo_store.connect()

    if delete:
        ids_to_remove = [d["_id"] for d, dupl in zip(data, duplicated) if dupl is not None]
        mongo_store.remove_docs({"_id": {"$in": ids_to_remove}})
    else:
        for d, dupl in zip(data, duplicated):
            if dupl is not None:
                mongo_store.collection.update({"_id": d["_id"]},{"$set": {"duplicated": data[dupl]["structure_id"]}})


def deduplicate_data_paral(n_procs: int, connection_data: dict, delete: bool = False,
                           stol: float = 0.03, primitive: bool = True, energy_tol: float = 0.03):
    """
    A function that marks the duplicated structures inside a mongodb collection.
    First looks for all the structures that do not have the "duplicated" value set (absent or None).
    Groups the data based on number of atoms and range of energies. For these sublists runs comparison
    with the StructureMatcher, chooses one of the structure as the reference and for other equivalent
    structures set a "duplicated" attribute in the document with value the structure_id of the reference
    structure.

    The energy ranges are determined by energy_tol and are simply splitting the energy in bins.
    As a consequence some structures may result as inequivalent even if they would match.
    This should not be a big issue.
    Args:
        n_procs: number of processes used in parallel.
        connection_data: a dictionary with the data that should be given to instantiate a MongoStore.
        delete: if True the duplicated data will be deleted instead of marked as duplicated.
        stol: the stol parameter passed to the StructureMatcher
        primitive: if True the match will be done after converting to primitive.
        energy_tol: the width of the bin used to divide the structures based on their energy per atom.
    """
    mongo_store = MongoStore(**connection_data)
    mongo_store.connect()

    fields = ["structure", "energy_per_atom", "nsites", "n_inequivalent_sites", "structure_id"]
    r = mongo_store.query({"duplicated": None}, properties=fields)
    if not r:
        return
    data = list(r)

    grouped_data = group_data(data, energy_tol=energy_tol)

    func = partial(run_deduplicate, connection_data=connection_data, stol=stol, primitive=primitive,
                   delete=delete)

    pool = multiprocessing.Pool(n_procs)
    pool.map(func, grouped_data)
    pool.close()
