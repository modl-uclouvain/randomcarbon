from abc import ABCMeta, abstractmethod
from monty.json import MSONable
from pymatgen.core.structure import Structure
from randomcarbon.utils.structure import get_properties, get_property, set_properties, get_struc_min_dist


class Tagger(MSONable, metaclass=ABCMeta):
    """
    A base class defining a way to set attributes in the final document
    to be stored in a Store.
    """

    @abstractmethod
    def tag(self, doc: dict, structure: Structure = None) -> dict:
        pass


class BasicTag(Tagger):

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        properties = get_properties(structure)
        if properties:
            doc.update(properties)

        doc["nsites"] = len(structure)

        return doc


class MetadataTag(Tagger):
    """
    Adds informations to the main documents or a list of tags
    """

    def __init__(self, info: dict = None, tags: list = None):
        self.info = info
        self.tags = tags

    def tag(self, doc: dict, structure: Structure = None) -> dict:
        doc.update(self.info)
        doc["tags"] = self.tags

        return doc




