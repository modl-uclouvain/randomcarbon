from randomcarbon.ga.initialization import RandomGenerator
from randomcarbon.evolution.evolvers.grow import AddSymmAtom
from randomcarbon.run.constraints import TemplateRepulsiveForce
from randomcarbon.utils.structure import get_property
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.kim.kim import KIM
from ase.ga.data import PrepareDB
from ase.ga import set_raw_score
from monty.serialization import dumpfn
from randomcarbon.utils.factory import Factory
from ase.spacegroup.symmetrize import FixSymmetry
import multiprocessing as mp
from randomcarbon.data import get_template
import numpy as np


aaa = AseAtomsAdaptor()

template = Structure.from_file(get_template("FAU.cif"))
spga = SpacegroupAnalyzer(template)
# optionally convert to primitive. Could be skipped if the primitive has the same size as the original.
template = spga.get_primitive_standard_structure()
spga = SpacegroupAnalyzer(template)

# store the templated and symmetry operations for later usage during the run.
template.to(filename="template.json")
symm_ops = spga.get_symmetry_operations()
dumpfn(symm_ops, "symm_ops.json")


def generate(seed):
    np.random.seed(seed)

    ev = AddSymmAtom(template=template, num_structures=1, symm_ops=symm_ops,
                     max_dist_eq=0.3, min_dist_current=1.2,
                     max_dist_current=1.6, min_dist_from_template=1.8,
                     max_dist_from_template=None, max_tests=3000)

    calc = KIM("Sim_LAMMPS_AIREBO_Morse_OConnorAndzelmRobbins_2015_CH__SM_460187474631_000")
    constraints = []
    # optionally add the repulsive force constraint
    # constraints += [TemplateRepulsiveForce(structure=template, sigma=0.4, height=1)]
    constraints += [Factory(FixSymmetry, set_atoms=True)]

    # generates relaxed structures with 3 to 5 inequivalent atoms.
    rg = RandomGenerator([3, 5], ev, calculator=calc, constraints=constraints, template=template,
                         min_dist_template=1.7, max_tests=50)

    s = rg.get_new_individual()

    if not s:
        print(f"failed to generate structure with seed {seed}")
        return None

    return s, get_property(s, "energy") / len(s)


# provides random seed for each of the processes and determines how many structures
# will be generated. To generate different structure use different ways to set the seeds.
seeds = list(range(20))

pool = mp.Pool(20)

results = pool.map(generate, seeds)

pool.close()

db_file = 'gadb.db'
d = PrepareDB(db_file_name=db_file)

for r in results:
    if r:
        s, energy = r
        a = aaa.get_atoms(s)
        # use the unrelaxed candidate, so that all the properties will be added correctly later.
        # the structure is already relaxed and getting the energy is fast anyway.
        # Alternatively raw_score, calculator and energy should be set (potentially a fingerprint)
        d.add_unrelaxed_candidate(a)
