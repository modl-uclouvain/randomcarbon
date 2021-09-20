from typing import List, Union
from randomcarbon.output.taggers.core import Tagger, BasicTag, MetadataTag
from randomcarbon.output.taggers.structure import SymmetryTag, MinDistTemplateTag, RingsStatsTag, NumNeighborsTag
from randomcarbon.output.taggers.calc import EnergyTag, CalculationInfoTag, EvolutionInfoTag

from randomcarbon.utils.factory import Factory
from randomcarbon.evolution.core import Evolver, Filter, Condition
from pymatgen.core.structure import Structure


__all__ = ["Tagger", "BasicTag", "MetadataTag", "SymmetryTag", "MinDistTemplateTag", "EnergyTag",
           "CalculationInfoTag", "EvolutionInfoTag", "RingsStatsTag", "NumNeighborsTag", "get_basic_taggers",
           "get_calc_taggers"]


def get_basic_taggers(template: Structure = None, info: dict = None, tags: list = None,
                      calculator: Factory = None, constraints: list = None):
    taggers = [BasicTag()]

    if info or tags:
        taggers.append(MetadataTag(info=info, tags=tags))

    if template:
        taggers.append(MinDistTemplateTag(template=template))

    taggers.append(EnergyTag(calculator=calculator, constraints=constraints))

    taggers.append(SymmetryTag())

    return taggers


def get_calc_taggers(calculator: Factory = None, constraints: list = None, optimizer: str = None,
                     fmax: float = None, evolvers: List[Union[Evolver, List]] = None,
                     blockers: List[Condition] = None, filters: List[Filter] = None):

    taggers = []

    if calculator or optimizer:
        taggers.append(CalculationInfoTag(calculator=calculator, constraints=constraints,
                                          optimizer=optimizer, fmax=fmax))
    if evolvers or blockers or filters:
        taggers.append(EvolutionInfoTag(evolvers=evolvers, blockers=blockers, filters=filters))

    return taggers
