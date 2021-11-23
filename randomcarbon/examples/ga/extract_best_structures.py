from ase.db import connect
from ase.io import write
from ase.ga.standard_comparators import SequentialComparator, EnergyComparator
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from monty.serialization import loadfn
from monty.os import cd, makedirs_p
from randomcarbon.ga.ase import LinearSumComparator, NumAtomsComparator, EnergyPerAtomComparator


min_score = 7.0
store_conventional = True

# work with a list of db files, in case structures have to be extracted from several runs.
db_paths = ["gadb.db"]
stored = []

template = loadfn("template.json")

# comp_en = EnergyComparator(dE=0.05)
comp_en = EnergyPerAtomComparator(dE=0.05)
comp_num_at = NumAtomsComparator()
comp_lc = LinearSumComparator(template.lattice, 0.3)
comp = SequentialComparator([comp_num_at, comp_en, comp_lc], [0, 0, 0])

for idbp, dbp in enumerate(db_paths):
    db = connect(dbp)

    entries = list(db.select(f'relaxed=1,raw_score>{min_score}', sort="-raw_score"))
    for i, e in enumerate(entries):
        a = e.toatoms()
        score = e.raw_score
        for (aa, ss) in stored:
            if comp.looks_like(a, aa):
                break
        else:
            stored.append((a, score))

makedirs_p("best_structures")
with cd("best_structures"):
    for i, (a, score) in enumerate(sorted(stored, key=lambda x: x[1], reverse=True)):
        # print(score)
        write(f"{i}_{score:.3f}_POSCAR.vasp", a)
        if store_conventional:
            s = AseAtomsAdaptor().get_structure(a)
            spga = SpacegroupAnalyzer(s)
            c = spga.get_conventional_standard_structure()
            c.to(filename=f"{i}_{score:.3f}_conv_POSCAR.vasp")
