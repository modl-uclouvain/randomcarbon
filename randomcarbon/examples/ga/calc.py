# based on calc.py from ASE examples.
# introduces the symmetry constraints and tunes the final score to the case with template
from randomcarbon.run.constraints import TemplateRepulsiveForce
from randomcarbon.utils.structure import get_struc_min_dist
from pymatgen.core import Structure
from ase.calculators.kim.kim import KIM
from ase.io import read, write
import sys
from ase.optimize import BFGS
from ase.spacegroup.symmetrize import FixSymmetry
from pymatgen.io.ase import AseAtomsAdaptor


fname = sys.argv[1]

a = read(fname)

t = Structure.from_file("template.json")

calc = KIM("Sim_LAMMPS_AIREBO_Morse_OConnorAndzelmRobbins_2015_CH__SM_460187474631_000")
constraints = []
constraints += [TemplateRepulsiveForce(structure=t, sigma=0.5, height=1)]
# NB in some tricky cases this may fail.
constraints += [FixSymmetry(a)]

a.calc = calc
a.set_constraint(constraints)
dyn = BFGS(a, trajectory=None, logfile=None)
dyn.run(fmax=0.05, steps=1000)

s_relax = AseAtomsAdaptor().get_structure(a)

min_dist = get_struc_min_dist(s_relax, t)

# if the structure is too close set a low score to exclude the structure.
if min_dist > 1.5:
    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy() / len(a)
else:
    a.info['key_value_pairs']['raw_score'] = -1

write(fname[:-5] + '_done.traj', a)
