from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.ga.data import PrepareDB
from monty.serialization import dumpfn
from randomcarbon.data import get_template
from randomcarbon.output.store import MongoStore

aaa = AseAtomsAdaptor()

# the generation of the json file is neede. It should be the same used for the
# data generated in the DB.
template = Structure.from_file(get_template("FAU.cif"))
spga = SpacegroupAnalyzer(template)
# optionally convert to primitive. Could be skipped if the primitive has the same size as the original.
template = spga.get_primitive_standard_structure()
spga = SpacegroupAnalyzer(template)

# store the template and symmetry operations for later usage during the run.
template.to(filename="template.json")
symm_ops = spga.get_symmetry_operations()
dumpfn(symm_ops, "symm_ops.json")

store = MongoStore(host='host', port=27017, username='username', password='password',
                   database='database_name', collection_name="collection_name")
store.connect()

n_best_structures = 10
n_random_structures = 10

# get a set of the best structures
# consider additional criteria here and below if multiple spacegroups have been generated in the same collection
criteria_best = {
    "energy_per_atom": {"$lte": -7.0},
    "template_min_dist": {"$gte": 1.6},
    "block_msg": None,
    "duplicated": None,  # preferably deduplicate the structures to avoid using the same structure multiple times
}

best_structures = []
for r in store.query(criteria=criteria_best, properties=["structure"], sort={"energy_per_atom": -1},
                           limit=n_best_structures):
    best_structures.append(Structure.from_dict(r["structure"]))

if len(best_structures) < n_best_structures:
    print(f"WARNING, only found {len(best_structures)}, queried {n_best_structures}.")

# now get some random structures
criteria_random = {
    "energy_per_atom": {"$lte": -6.0},
    "template_min_dist": {"$gte": 1.6},
    "block_msg": None,
    "duplicated": None,  # preferably deduplicate the structures to avoid using the same structure multiple times
}

results = store.collection.aggregate([
    {"$match": criteria_random},
    {"$sample": {"size": n_random_structures}}
])

random_structures = []
for r in results:
    random_structures.append(Structure.from_dict(r["structure"]))

if len(random_structures) < n_random_structures:
    print(f"WARNING, only found {len(random_structures)}, queried {n_random_structures}.")


db_file = 'gadb.db'
d = PrepareDB(db_file_name=db_file)

for s in best_structures + random_structures:
    a = aaa.get_atoms(s)
    # use the unrelaxed candidate, so that all the properties will be added correctly later.
    # the structure is already relaxed and getting the energy is fast anyway.
    # Alternatively raw_score, calculator and energy should be set (potentially a fingerprint)
    d.add_unrelaxed_candidate(a)
