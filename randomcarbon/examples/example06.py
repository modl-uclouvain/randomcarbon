import logging
from randomcarbon.utils.structure import add_new_symmetrized_atom
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CutOffDictNN
from randomcarbon.run.runners import BranchingParallelRunner
from randomcarbon.evolution.evolvers.grow import AddSymmAtom
from randomcarbon.evolution.blockers.structure import MinTemplateDistance
from randomcarbon.evolution.blockers.energy import EnergyAtoms
from randomcarbon.utils.factory import Factory
from randomcarbon.evolution.filters.limit import MatchingStructures
from randomcarbon.data import get_template
from randomcarbon.output.store import MongoStore
from randomcarbon.output.taggers import get_basic_taggers, get_calc_taggers, RingsStatsTag, NumNeighborsTag
from ase.calculators.kim.kim import KIM
from ase.spacegroup.symmetrize import FixSymmetry


logging.basicConfig(format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
logger = logging.getLogger("randomcarbon")
logger.setLevel(logging.INFO)

template = Structure.from_file(get_template("FAU_zeolite_conv_std.cif"))

# convert to primitive and work directly with its version
spga = SpacegroupAnalyzer(template)
template = spga.get_primitive_standard_structure()

# this needs the symmetry operations of the primitive that has been generated
spga = SpacegroupAnalyzer(template, symprec=0.001)
symm_ops = spga.get_symmetry_operations()


# generate 5 initial structures to add to the queue
initial_structures = [add_new_symmetrized_atom(template=template) for _ in range(5)]

constraints = [Factory(callable=FixSymmetry, set_atoms=True)]
# calculator = Factory(callable=KIM, model_name="Sim_LAMMPS_AIREBO_Morse_OConnorAndzelmRobbins_2015_CH__SM_460187474631_000")
calculator = Factory(callable=KIM, model_name="Tersoff_LAMMPS_Tersoff_1989_SiC__MO_171585019474_002")

evolvers = [AddSymmAtom(template=template, num_structures=10, max_tests=100, symm_ops=symm_ops)]
sm = StructureMatcher(primitive_cell=False, stol=0.001, scale=False)
filters = [MatchingStructures(sm)]
blockers = [MinTemplateDistance(template, min_dist=3),
            EnergyAtoms(criteria={600: -4, 700: -4.5, 800: -5}, calculator=calculator, constraints=constraints)]
taggers = get_basic_taggers(template=template, info={"run_name": "example3", "zeolite": "FAU"},
                            calculator=calculator, constraints=constraints)
taggers += get_calc_taggers(calculator=calculator, optimizer="BFGS", fmax=0.05)
taggers.append(RingsStatsTag(method=5, maximum_search_depth=5, cutoff_rad={("C", "C"): 1.9},
                             executable="/path/to/rings/executable"))
taggers.append(NumNeighborsTag(CutOffDictNN({("C", "C"): 1.9})))

store = MongoStore(database="database_name", collection_name="collection_name", host="host.com", port=27017,
                   username="username", password="password")

runner = BranchingParallelRunner(calculator_factory=calculator, evolvers=evolvers,
                                 initial_structures=initial_structures, blockers=blockers, filters=filters,
                                 fmax=0.05, steps=1000, constraints=constraints, optimizer="BFGS",
                                 opt_kwargs={"logfile": None}, store=store, taggers=taggers)

runner.run(nprocs=2, max_structures=30)
