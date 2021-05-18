import logging
from typing import Union
from collections import defaultdict
import numpy as np
import networkx as nx
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import NearNeighbors, CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from randomcarbon.output.taggers.core import Tagger
from randomcarbon.utils.structure import get_properties, get_property, set_properties, get_struc_min_dist
from randomcarbon.utils.factory import Factory
from randomcarbon.rings.input import RingMethod, RingsInput
from randomcarbon.rings.run import run_rings

logger = logging.getLogger(__name__)


class MinDistTemplateTag(Tagger):
    """
    Adds the minimum distance of the structure from the template
    """

    def __init__(self, template: Structure):
        self.template = template

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        if structure is None:
            raise ValueError("The structure should be defined")

        doc["template_min_dist"] = get_struc_min_dist(self.template, structure)

        return doc


class SymmetryTag(Tagger):
    """
    Sets the calculated spacegroup as obtained from a given SpaceGroupAnalyzer.
    """

    def __init__(self, symprec: float = 0.01, angle_tolerance: float = 5.0, full=False):
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance
        self.full = full

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        if structure is None:
            raise ValueError("The structure should be defined")

        spga = SpacegroupAnalyzer(structure=structure, symprec=self.symprec,
                                  angle_tolerance=self.angle_tolerance)

        sym_struct = spga.get_symmetrized_structure()

        doc["n_equivalent_sites"] = [len(l) for l in sym_struct.equivalent_sites]
        doc["n_inequivalent_sites"] = len(sym_struct.equivalent_sites)

        if self.full:
            d_spg = dict(
                crystal_system = spga.get_crystal_system(),
                hall = spga.get_hall(),
                number = spga.get_space_group_number(),
                selfsymbol = spga.get_space_group_symbol()
            )
            doc.update(d_spg)
        else:
            doc["spgn"] = spga.get_space_group_number()

        return doc


class NumNeighborsTag(Tagger):

    def __init__(self, nn: Union[Factory, NearNeighbors] = None):
        """
        Args:
            nn: a NearNeighbors, or, if needed, a Factory that contains it.
                May be needed since NearNeighbors is not MSONable.
                If None it will be used CrystalNN with default parameters.
        """
        self.nn = nn

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        if structure is None:
            raise ValueError("The structure should be defined")

        nn = self.nn
        if not nn:
            nn = CrystalNN()
        if isinstance(nn, Factory):
            nn = nn.generate()

        d = defaultdict(int)
        for i in range(len(structure)):
            d[nn.get_cn(structure, i)] += 1

        doc["neighbors_number"] = dict(d)

        return doc


class NeighborsStatsTag(Tagger):

    def __init__(self, nn: Union[Factory, NearNeighbors] = None,
                 symprec: float = 0.01, angle_tolerance: float = 5.0):
        """
        Args:
            nn: a NearNeighbors, or, if needed, a Factory that contains it.
                May be needed since NearNeighbors is not MSONable.
                If None it will be used CrystalNN with default parameters.
            symprec: the symprec value used in spglib to determine the
                spacegroup of the system and the symmetry equivalent atoms.
            angle_tolerance: the angle_tolerance value used in spglib to determine the
                spacegroup of the system and the symmetry equivalent atoms.
        """
        self.nn = nn
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        if structure is None:
            raise ValueError("The structure should be defined")

        nn = self.nn
        if not nn:
            nn = CrystalNN()
        if isinstance(nn, Factory):
            nn = nn.generate()

        all_nn_info = nn.get_all_nn_info(structure)
        d_cn = defaultdict(int)

        if self.symprec is not None and self.angle_tolerance is not None:
            spga = SpacegroupAnalyzer(structure, symprec=self.symprec, angle_tolerance=self.angle_tolerance)
            sym_struct = spga.get_symmetrized_structure()
            indices = [l[0] for l in sym_struct.equivalent_sites]
            n_equivalent_sites = [len(l) for l in sym_struct.equivalent_sites]
        else:
            indices = list(range(len(structure)))
            n_equivalent_sites = [1] * len(structure)

        avg_dist_per_site = []
        for i, n_eq, nn_info in zip(indices, n_equivalent_sites, all_nn_info):
            d_cn[len(nn_info)] += 1 * n_eq
            avg_dist = 0
            for dict_nn in nn_info:
                avg_dist += structure.get_distance(i, dict_nn["site_index"])
            avg_dist /= len(nn_info)

        full_list = np.concatenate([[av] * n_av for av, n_av in zip(avg_dist_per_site, n_equivalent_sites)])
        global_avg_dist = full_list.mean()
        global_std_dev = full_list.std()

        doc["neighbors_data"] = {
            "neighbors_number": dict(d_cn),
            "avg_dist_per_site": avg_dist_per_site,
            "avg_dist": global_avg_dist,
            "dist_std_dev": global_std_dev
        }

        return doc


class RingsStatsTag(Tagger):

    def __init__(self, method: Union[RingMethod, int], lattice_matrix: bool = True,
                 maximum_search_depth: int = 5, cutoff_rad: Union[dict, NearNeighbors] = None,
                 grmax: float = None, executable: str = "rings", irreducible: bool = True):
        self.method = method
        self.lattice_matrix = lattice_matrix
        self.maximum_search_depth = maximum_search_depth
        self.cutoff_rad = cutoff_rad
        self.grmax = grmax
        self.executable = executable
        self.irreducible = irreducible

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        
        inp = RingsInput(structure=structure, methods=[self.method], lattice_matrix=self.lattice_matrix,
                            maximum_search_depth=self.maximum_search_depth, cutoff_rad=self.cutoff_rad,
                            grmax=self.grmax)
        if get_property(structure, 'rings') is not None and get_property(structure, 'rings')['rings_input'] == inp:
            
            '''
            I'm not sure if it is a good way to compare the two input parameters
            '''

            doc['rings_stats'] = get_property(structure, 'rings')['stats']

            return doc           
            
            
            
        else:
            
            out = run_rings(inp, executable=self.executable, irreducible=self.irreducible)
            
            if not out:
                sid = get_property(structure, "structure_id")
                logger.warning(f"no output produced by rings for structure {sid}")
                return doc

            rings_list = out[self.method]
            
            doc["ring_stats"] = rings_list.get_stats_dict()

            return doc
            




class ConnectedTag(Tagger):
    """
    Sets a "connected" key in the doc to True if all the atoms are connected,
    based on a NearNeighbors and a StructureGraph
    """
    def __init__(self, nn: Union[Factory, NearNeighbors] = None,
                 supercell: bool = True):
        """
        Args:
            nn: a NearNeighbors, or, if needed, a Factory that contains it.
                May be needed since NearNeighbors is not MSONable.
                If None it will be used CrystalNN with default parameters.

        """
        self.nn = nn
        self.supercell = supercell

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        if structure is None:
            raise ValueError("The structure should be defined")

        nn = self.nn
        if not nn:
            nn = CrystalNN()
        if isinstance(nn, Factory):
            nn = nn.generate()

        if self.supercell:
            structure = structure.copy()
            structure.make_supercell(2)

        sg = StructureGraph.with_local_env_strategy(structure=structure, strategy=nn)

        # need to create a undirected graph
        g = nx.Graph(sg.graph)

        doc["connected"] = nx.is_connected(g)

        return doc
