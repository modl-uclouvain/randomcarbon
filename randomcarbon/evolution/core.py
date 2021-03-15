"""
Main module for the definition and execution of the evolution of a structure.
Defines the base classes and provides functions to evolve the structure.
"""
import random
import logging
from abc import ABCMeta, abstractmethod
from pymatgen.core.structure import Structure
from typing import List, Union, Optional
from monty.json import MSONable
from randomcarbon.utils.structure import set_structure_id, set_properties, get_property

logger = logging.getLogger(__name__)


class Evolver(MSONable, metaclass=ABCMeta):
    """
    Base class for the objects that define how to evolve a structure.
    This can include modifying the atoms currently present in the cell.

    When needing object like constraint or Calculators, subclasses should
    use the Factory object to wrap them. This should prevent problems
    like potential race conditions, storage of partial results
    in the objects between different runs and the objects not being
    MSONable.
    """

    @abstractmethod
    def evolve(self, structure: Structure) -> List[Structure]:
        """
        The method that performs the modification of the structure.
        Should return a list of modified structures, if only one
        structure is generated should be a list with one structure.
        If some structures are favorite they should be returned
        sorted in decreasing order of interest. The caller may decide
        to only take some of the structures into account.
        The returned list can be empty if no structure could be
        generated.

        Args:
            structure: a pymatgen structure that should be modified

        Returns:
            List[Structure]: list of modified structures. If meaningful
                sorted in decreasing order of preference for new structures
                to be picked.
        """
        pass


class Filter(MSONable, metaclass=ABCMeta):
    """
    Base class for the objects that define how to filter or rearrange
    a list of structures.

    When needing object like constraint or Calculators, subclasses should
    use the Factory object to wrap them. This should prevent problems
    like potential race conditions, storage of partial results
    in the objects between different runs and the objects not being
    MSONable.
    """
    @abstractmethod
    def filter(self, structures: List[Structure]) -> List[Structure]:
        """
        The method that performs the filtering or rearrangement.
        If all the structures are eliminated, should return an empty list.

        Args:
            structures: the list of Structures to be modified.

        Returns:
            A filtered list of structures.
        """
        pass


class Blocker(MSONable, metaclass=ABCMeta):
    """
    Base class for the object that will analyze a structure an halt the
    evolution based on some criteria, before the generation of new structures.

    When needing object like constraint or Calculators, subclasses should
    use the Factory object to wrap them. This should prevent problems
    like potential race conditions, storage of partial results
    in the objects between different runs and the objects not being
    MSONable.
    """

    @abstractmethod
    def block(self, structure: Structure) -> Optional[str]:
        """
        Checks if the passed structure fullfills the criteria to be
        blocked.

        Args:
            structure: a Structure to be checked.

        Returns:
            str: if present signals that the evolution should be blocked and
                contains a short description of the reason. If None the evolution
                can continue.
        """
        pass


def evolve_structure(
        structure: Structure,
        evolvers: List[Union[Evolver, List]],
        blockers: List[Blocker] = None,
        filters: List[Filter] = None
) -> List[Structure]:
    """
    Performs the evolution of a structure.
    Applies evolvers and filters and returns a list of structures.
    Can return an empty list if the incoming structure fullfills the
    criteria defined in the blockers.

    Evolvers can be provided either as a simple list or a list of
    lists, where each sublists should be of the form [Evolver, float].
    The float should be a number from 0 to 1 and represent the probability
    that the evolver will be applied. Note that the, if all the probabilities
    are lower than 1, it could also happen that no evolver will be applied.

    The generated structures will be added an "structure_id" and a "history" in the
    properties attribute, based on the history of the incoming structure.

    Args:
        structure (Structure): the structure used as base for the evolution.
        evolvers (list of Evolver): a list of Evolvers to be used to generate
            new structures.
        blockers (list of Blocker): determine if no further evolution should be
            done on the incoming structure.
        filters (list of Filter): a list of Filters that will reduce the amount
            of structures if needed.

    Returns:
        list(Structure): a list of generated structures. An empty list if no
            further evolution is possible according to the input objects.
    """
    if blockers:
        for b in blockers:
            block_msg = b.block(structure)
            if block_msg:
                logger.info(f"Stopping evolution of structure {get_property(structure, 'structure_id')}: {block_msg}")
                set_properties(structure, {"block_msg": block_msg})
                return []

    structures = []
    for e in evolvers:
        # if it is a list/tuple the second element represent the
        # probability with which the evolver should be used.
        if isinstance(e, (list, tuple)):
            prob = e[1]
            if random.random() > prob:
                continue
            e = e[0]
        structures.extend(e.evolve(structure))

    if structures and filters:
        for f in filters:
            structures = f.filter(structures)

    history = get_property(structure, "history")
    if not history:
        history = []
    else:
        history = list(history)
    sid = get_property(structure, "structure_id")
    if sid:
        history.append(sid)
    for s in structures:
        set_structure_id(s)
        set_properties(s, {"history": history})

    return structures
