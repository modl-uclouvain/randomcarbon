# example script for running rings analysis on a structure as a post processing
from randomcarbon.rings.run import run_rings
from randomcarbon.rings.input import RingsInput, RingMethod
from pymatgen import Structure

s = Structure.from_file("structure.cif")

# generate the input required to run rings.
# the list of methods correspond to the different algorithms implemented. See
# http://rings-code.sourceforge.net/index.php?option=com_content&view=article&id=56%3Ainput-file&catid=37%3Ausing-rings&Itemid=18
# http://rings-code.sourceforge.net/index.php?option=com_content&view=article&id=45%3Aring-statistics&catid=36%3Aphy-chi&Itemid=53
# Note that the maximum search depth corresponds to the definition in rings, i.e half of the maximum search
# depth actually used.
# cutoff_rad is the maximum distance allowed between two atoms to be considered connected.
# Here a simple choice for carbon structures is given. If not defined not defined
# will be determined automatically using pymatgen. This could be expensive if done repeatedly and
# it is better to define it explicitly for carbon structures.
inp = RingsInput(structure=s, methods=[RingMethod.KING_HOMOPOLAR, RingMethod.PRIMITIVE],
                 maximum_search_depth=4, cutoff_rad={("C", "C"): 1.9})

# the output will be a dictionary with two items (since two RingMethod were passed to the input). Each item
# is a RingsList with the rings list and some statistics extracted from the rings code.
# Keys are the RingMethod Enum. The Enum values are integers as defined in the rings code (e.g. PRIMITIVE corresponds
# to 5)
out_dict = run_rings(rings_input=inp, executable="/path/to/rings/executable")

rings_list = out_dict[RingMethod.PRIMITIVE]
print(rings_list.get_dataframe())

# prints a script that can be copied to the jmol script shell to plot the rings
# with a color depending on the size of the ring.
# inside a jupyter notebook it is possible to call show_jsmol() to plot the rings with jupyter_jsmol
print(rings_list.jmol_script())

# alternatively it is possible to visualize the atoms of the rings by generating a structure with
# atoms with a different specie for a specific ring. It can then be opened with any viewer.
s_ring = rings_list.structure_with_ring_specie(ring_ind=0, specie="Si")
s_ring.to(filename="Si_ring.cif")
