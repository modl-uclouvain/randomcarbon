"""
logical conditions. based on other conditions
"""
import logging
from typing import List, Optional, Tuple, Union
from randomcarbon.evolution.core import Condition
from pymatgen.core.structure import Structure


logger = logging.getLogger(__name__)


class Not(Condition):

    def __init__(self, condition: Condition):
        self.condition = condition

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:
        v, msg_cond = self.condition.satisfied(structure)
        if msg_cond is None:
            msg = f"NOT condition {self.condition.__name__}"
        else:
            msg = f"NOT: {msg_cond}"

        return not v, msg


class Or(Condition):

    def __init__(self, conditions: Union[Condition, List[Condition]]):
        if not isinstance(conditions, (tuple, list)):
            conditions = [conditions]

        self.conditions = conditions

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:
        msg_list = []
        for c in self.conditions:
            v, msg_cond = c.satisfied(structure)
            if msg_cond is None:
                msg = f"condition {c.__name__} is {v}"
            else:
                msg = msg_cond
            msg_list.append(msg)
            if v:
                return True, f"OR: {msg}"

        return False, "OR: None of the conditions are satisfied: " + " | ".join(msg_list)


class And(Condition):

    def __init__(self, conditions: Union[Condition, List[Condition]]):
        if not isinstance(conditions, (tuple, list)):
            conditions = [conditions]

        self.conditions = conditions

    def satisfied(self, structure: Structure) -> Tuple[bool, Optional[str]]:
        msg_list = []
        for c in self.conditions:
            v, msg_cond = c.satisfied(structure)
            if msg_cond is None:
                msg = f"condition {c.__name__} is {v}"
            else:
                msg = msg_cond
            msg_list.append(msg)
            if not v:
                return False, f"AND: {msg}"

        return True, "AND: all the conditions are satisfied: " + " | ".join(msg_list)
