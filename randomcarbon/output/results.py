from typing import List, Any
import datetime
import multiprocessing
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.serialization import MontyDecoder
from randomcarbon.output.store import Store
from randomcarbon.utils.structure import get_properties, set_structure_id, get_struc_min_dist
from randomcarbon.output.taggers.core import Tagger


def store_results(structure: Structure, store: Store, taggers: List[Tagger] = None):
    """
    Function that generates a results document and adds it to a Store.
    Args:
        structure: the Structure that should be stored.
        store: the Store used to put the data.
        template: the template Structure used to generate the final structure.
        taggers: a list of Tagger objects to add values to the stored document.
    """
    set_structure_id(structure)
    doc = {"structure": structure.as_dict(),
           "created_on": datetime.datetime.utcnow()}

    if taggers:
        for t in taggers:
            doc = t.tag(doc, structure=structure)

    store.insert(doc)


def store_results_old(structure: Structure, store: Store, template: Structure = None):
    """
    Function that generates a results document and adds it to a Store.
    Args:
        structure: the Structure that should be stored.
        store: the Store used to put the data.
        template: the template Structure used to generate the final structure.
    """
    set_structure_id(structure)
    properties = get_properties(structure)
    d = {"structure": structure.as_dict()}
    d.update(properties)
    if "energy" in d:
        d["energy_per_atom"] = d["energy"] / len(structure)

    if template:
        d["template_min_dist"] = get_struc_min_dist(structure, template)

    spga = SpacegroupAnalyzer(structure)
    sym_struct = spga.get_symmetrized_structure()

    d["n_equivalent_sites"] = [len(l) for l in sym_struct.equivalent_sites]
    d["nsites"] = len(structure)

    d["created_on"] = datetime.datetime.utcnow()

    store.insert(d)


def _set_tags(key_value: Any, store: Store, taggers: List[Tagger], structure_key: str = "structure"):
    doc = next(store.query(criteria={store.key: key_value}))
    s = Structure.from_dict(doc[structure_key])
    for t in taggers:
        doc = t.tag(doc, s)
    store.update(docs=doc, key=store.key)


def _run_set_tags(key_value: Any, store_dict: dict, taggers: List[Tagger],
                  structure_key: str = "structure"):
    store = MontyDecoder().process_decoded(store_dict)
    store.connect()
    _set_tags(key_value, store, taggers, structure_key)


def tag_results(store: Store, taggers: List[Tagger], criteria: dict = None, n_procs: int = 1,
                structure_key: str = "structure") -> int:

    store.connect()

    # query to get a list of ids. It will be processed afterwards.
    results = list(store.query(criteria=criteria, properties=[store.key]))
    if not results:
        return 0

    if n_procs > 1:
        pool = multiprocessing.Pool(n_procs)
        store_dict = store.as_dict()
        data = [(r[store.key], store_dict, taggers, structure_key) for r in results]
        pool.starmap(_run_set_tags, data)
        pool.close()
        pool.join()
    else:
        for r in results:
            _set_tags(r[store.key], store, taggers, structure_key)

    return len(results)
