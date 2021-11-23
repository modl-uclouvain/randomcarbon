from random import random
from ase.io import write
import time
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import SequentialComparator, EnergyComparator
from ase.ga.offspring_creator import OperationSelector
from ase.ga.parallellocalrun import ParallelLocalRun
from randomcarbon.ga.ase import MixInequivalent, EvolverWrapper, NumAtomsComparator
from randomcarbon.ga.ase import LinearSumComparator, EnergyPerAtomComparator
from randomcarbon.evolution.evolvers.grow import AddSymmAtom, AddSymmAtomBridge, AddSymmAtomUndercoord
from randomcarbon.evolution.evolvers.reduce import MergeAtoms, RemoveAtoms
from randomcarbon.evolution.evolvers.modify import MoveAtoms
from monty.serialization import loadfn
from monty.os import makedirs_p
from ase.ga.ofp_comparator import OFPComparator


population_size = 20
# probability that the evolution is a mutation, otherwise is a mix of 2 structures
mutation_probability = 0.7
# number of new individuals generated.
n_to_test = 1000

t = loadfn("template.json")
symm_ops = loadfn("symm_ops.json")

# Initialize the different components of the GA
da = DataConnection('gadb.db')
tmp_folder = 'tmp_folder/'
makedirs_p(tmp_folder)

# An extra object is needed to handle the parallel execution
parallel_local_run = ParallelLocalRun(data_connection=da,
                                      tmp_folder=tmp_folder,
                                      n_simul=25,
                                      calc_script='calc.py')

# it can be used the total energy, but in our case it is easier to define a threshold in the
# differences for the energy per atom.
# comp_en = EnergyComparator(dE=0.05)
comp_en = EnergyPerAtomComparator(dE=0.05)
comp_num_at = NumAtomsComparator()
# using the Oganov comparator is an option, but the LinearSumComparator seems faster and enough for our needs.
#ofp_comp = OFPComparator(dE=0.05)
#comp = SequentialComparator([comp_num_at, ofp_comp], [0,0])
comp_lc = LinearSumComparator(t.lattice, 0.3)
comp = SequentialComparator([comp_num_at, comp_en, comp_lc], [0,0,0])

# the definition of the mutation operators. one of the following is randomly selected at each step, based on
# the given probabilities.
add_at = EvolverWrapper(AddSymmAtom(template=t, symm_ops=symm_ops, min_dist_from_template=1.6, max_dist_eq=0.3))
add_at_undercoord = EvolverWrapper(AddSymmAtomUndercoord(template=t, symm_ops=symm_ops, min_dist_from_template=1.6, max_dist_eq=0.3))
add_at_undercoord2 = EvolverWrapper(AddSymmAtomUndercoord(template=t, symm_ops=symm_ops, min_dist_from_template=1.6, max_dist_eq=0.6))
merge_at = EvolverWrapper(MergeAtoms(symm_ops=symm_ops))
remove_at = EvolverWrapper(RemoveAtoms(symm_ops=symm_ops))
move_at = EvolverWrapper(MoveAtoms(symm_ops=symm_ops, min_displ=0.6, max_displ=1.6), min_dist=0.8, max_tests=100)
mutations = OperationSelector([0.5, 0.25, 0.25, 0.7, 0.7, 0.9],
                              [add_at, add_at_undercoord, add_at_undercoord2, merge_at, remove_at, move_at])

pairing = MixInequivalent(symm_ops=symm_ops)

# Relax all unrelaxed structures (e.g. the starting population)
while da.get_number_of_unrelaxed_candidates() > 0:
    a = da.get_an_unrelaxed_candidate()
    parallel_local_run.relax(a)

# Wait until the starting population is relaxed
while parallel_local_run.get_number_of_jobs_running() > 0:
    time.sleep(5.)

# create the population
population = Population(data_connection=da,
                        population_size=population_size,
                        comparator=comp)

# test n_to_test new candidates
for i in range(n_to_test):
    print('Now starting configuration number {0}'.format(i))

    # Check if we want to do a mutation or a pairing
    if random() < mutation_probability:
        a1= population.get_one_candidate()
        a3, desc = mutations.get_new_individual([a1])
    else:
        a1, a2 = population.get_two_candidates()
        a3, desc = pairing.get_new_individual([a1, a2])
    if a3 is None:
        continue
    da.add_unrelaxed_candidate(a3, description=desc)

    # Relax the new candidate
    parallel_local_run.relax(a3)
    population.update()

# Wait until the last candidates are relaxed
while parallel_local_run.get_number_of_jobs_running() > 0:
    time.sleep(5.)

write('all_candidates.traj', da.get_all_relaxed_candidates())
