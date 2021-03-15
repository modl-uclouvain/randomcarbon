from typing import Union, List
import os
import itertools
import logging
from enum import IntEnum
from monty.os import makedirs_p, cd
from monty.json import MSONable
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import NearNeighbors, CrystalNN
from ase.io import write
from randomcarbon.utils.bonding import get_pairs_max_dist

logger = logging.getLogger(__name__)


class RingMethod(IntEnum):
    """
    Enumerates the possible algorithms that can be used by the rings code.
    The integer number matches the one used by rings in the output to identify the different methods.
    """

    CLOSED_PATHS = 0
    KING_NOT_HOMOPOLAR = 1
    GUTTMAN_NOT_HOMOPOLAR = 2
    KING_HOMOPOLAR = 3
    GUTTMAN_HOMOPOLAR = 4
    PRIMITIVE = 5
    STRONG = 6

    @classmethod
    def from_dict(cls, d):
        return cls(d["method"])

    def as_dict(self):
        return {"@module": self.__class__.__module__,
                "@class": self.__class__.__name__,
                "method": self.value}


class RingsInput(MSONable):
    """
    An object to describe and write the input for a rings analysis.
    """

    def __init__(self, structure: Structure, methods: List[Union[RingMethod, int]] = None, lattice_matrix: bool = True,
                 maximum_search_depth: int = 5, traj_file_name: str = "structure.xyz",
                 cutoff_rad: Union[dict, NearNeighbors] = None, grmax: float = None):
        self.structure = structure
        if not methods:
            methods = [RingMethod.PRIMITIVE]
        methods = [RingMethod(m) for m in methods]
        self.methods = methods
        self.lattice_matrix = lattice_matrix
        self.maximum_search_depth = maximum_search_depth
        self.traj_file_name = traj_file_name

        if cutoff_rad is None:
            cutoff_rad = CrystalNN()

        if isinstance(cutoff_rad, NearNeighbors):
            cutoff_rad = get_pairs_max_dist(structure=self.structure, near_neighbors=cutoff_rad, pad=0.01)

        el_amt_dict = self.structure.composition.get_el_amt_dict()
        for a1, a2 in itertools.combinations_with_replacement(el_amt_dict.keys(), 2):
            k_pair = tuple(sorted((a1, a2)))
            if k_pair not in cutoff_rad:
                cutoff_rad[k_pair] = 0.

        self.cutoff_rad = cutoff_rad

        if grmax is None:
            grmax = max(self.cutoff_rad.values())
        self.grmax = grmax

    def get_input_string(self):
        significant_figures = 6
        format_str = "{{:.{0}f}}".format(significant_figures)

        el_amt_dict = self.structure.composition.get_el_amt_dict()

        input_lines = ["#######################################",
                       "#       R.I.N.G.S. input file         #",
                       "#######################################"]

        input_lines.append(f"input generated for {self.structure.composition.reduced_formula}")
        input_lines.append(f"{len(self.structure)}")
        input_lines.append(f"{len(el_amt_dict)}")
        input_lines.append(" ".join(el_amt_dict.keys()))  # use the structure
        input_lines.append(f"1")  # number of configurations. in principle only one
        if self.lattice_matrix:
            input_lines.append("1")  # format of the lattice. 1 to be consistent with the following.
            for v in self.structure.lattice.matrix:
                input_lines.append(" ".join(format_str.format(vi) for vi in v))
        else:
            input_lines.append("0")  # format of the lattice. 1 to be consistent with the following.
            input_lines.append(" ".join(format_str.format(a) for a in self.structure.lattice.abc))
            input_lines.append(" ".join(format_str.format(a) for a in self.structure.lattice.angles))
        input_lines.append("1.0")  # Integration time step t of the Newton's equations of motion. not needed here
        input_lines.append("ANI")  # xyz format
        input_lines.append(self.traj_file_name)  # vasp format

        input_lines.append("200")  # Real space discretization for the g(r) calculations.
        input_lines.append("500")  # Reciprocal space discretization for the S(q) calculations.
        input_lines.append("25")  # Maximum modulus of the reciprocal space vectors for the S(q) calculations.
        input_lines.append(
            "0.125")  # Smoothing factor for the S(q) calculated using the sampling of vectors from the reciprocal space.
        input_lines.append(
            "90")  # Angular discretization between [0-180] degrees used to compute the angle distribution.
        input_lines.append("20")  # Real space discretization for the voids and for the ring statistics.
        input_lines.append(f"{self.maximum_search_depth}")  # (Maximum search depth)/2 for the ring statistics - (min = 2)
        input_lines.append("2")  # Maximum search depth for the chain statistics - (min = 2)

        input_lines.append("#######################################")

        for k_pair, dist in self.cutoff_rad.items():
            input_lines.append(f"{k_pair[0]} {k_pair[1]} " + format_str.format(dist))

        input_lines.append(f"Grtot " + format_str.format(self.grmax))

        input_lines.append("#######################################")

        return "\n".join(input_lines)

    def get_options_string(self):

        #NB: the number of lines seems to be important. Even those wil dashes should be there. otherwise input pasing fails
        options_lines = ["#######################################",
                         "        R.I.N.G.S. options file       #",
                         "#######################################"]
        options_lines.append("PBC             .true.")
        options_lines.append("Frac             .false.")
        options_lines.append("g(r)             .false.")
        options_lines.append("S(q)             .false.")
        options_lines.append("S(k)             .false.")
        options_lines.append("gfft(r)             .false.")
        options_lines.append("MSD             .false.")
        options_lines.append("atMSD             .false.")
        options_lines.append("Bonds             .true.")
        options_lines.append("Angles             .false.")
        options_lines.append("Chains             .false.")

        options_lines.append("---- ! Chain statistics options ! -----")
        options_lines.append("Species             0")
        options_lines.append("AAAA             .false.")
        options_lines.append("ABAB             .false.")
        options_lines.append("1221             .false.")

        options_lines.append("---------------------------------------")
        options_lines.append("Rings             .true.")

        options_lines.append("---- ! Ring statistics options ! -----")
        options_lines.append("Species             0")
        options_lines.append("ABAB             .false.")
        options_lines.append(f"Rings0             .{RingMethod.CLOSED_PATHS in self.methods}.")  # all closed paths in the box.
        options_lines.append(f"Rings1             .{RingMethod.KING_NOT_HOMOPOLAR in self.methods}.")  # King's shortest path rings - homopolar bond(s) do not affect the search.
        options_lines.append(f"Rings2             .{RingMethod.GUTTMAN_NOT_HOMOPOLAR in self.methods}.")  #  Guttman's shortest path rings - homopolar bond(s) do not affect the search.
        options_lines.append(f"Rings3             .{RingMethod.KING_HOMOPOLAR in self.methods}.")  #  King's shortest path rings - homopolar bond(s) can shortcut the rings.
        options_lines.append(f"Rings4             .{RingMethod.GUTTMAN_HOMOPOLAR in self.methods}.")  # Guttman's shortest path rings - homopolar bond(s) can shortcut the rings.
        options_lines.append(f"Prim_Rings         .{RingMethod.PRIMITIVE in self.methods}.")
        options_lines.append(f"Str_Rings          .{RingMethod.STRONG in self.methods}.")
        options_lines.append("BarycRings          .false.")
        options_lines.append("Prop-1             .false.")
        options_lines.append("Prop-2             .false.")
        options_lines.append("Prop-3             .false.")
        options_lines.append("Prop-4             .false.")
        options_lines.append("Prop-5             .false.")

        options_lines.append("---------------------------------------")
        options_lines.append("Vacuum             .false.")
        options_lines.append("#######################################")
        options_lines.append("         Outputting options           #")
        options_lines.append("#######################################")
        options_lines.append("Evol             .false.")
        options_lines.append("Dxout             .false.")
        options_lines.append("! OpenDX visualization options !  --")
        options_lines.append("RadOut             .false.")
        options_lines.append("RingsOut             .false.")
        options_lines.append("DRngOut         .false.")
        options_lines.append("VoidsOut             .false.")
        options_lines.append("TetraOut             .false.")
        options_lines.append("TrajOut             .false.")
        options_lines.append("---------------------------------------")
        options_lines.append("Output        rings.out")
        options_lines.append("#######################################")

        return "\n".join(options_lines)

    def write(self, workdir, input_filename="ring_input"):

        input_str = self.get_input_string()
        options_str = self.get_options_string()

        makedirs_p(workdir)
        with cd(workdir):
            # standard name for the input data folder of the RINGS code.
            data_dir = "data"

            makedirs_p(data_dir)

            # use ase since pymatgen does not support xyz format for structure
            aaa = AseAtomsAdaptor()
            atoms = aaa.get_atoms(self.structure)

            traj_path = os.path.join(data_dir, self.traj_file_name)
            write(traj_path, atoms, format="xyz")

            with open(input_filename, "wt") as f:
                f.write(input_str)

            with open("options", "wt") as f:
                f.write(options_str)
