from typing import Callable
from monty.json import MSONable, MontyDecoder
from ase.atoms import Atoms
import ase.optimize


class Factory(MSONable):
    """
    Wrapper class for existing classes.
    Allows to generate multiple instances of the chosen class based
    on the arguments passed to the __init__.
    Additionally provides an option for MSON serialization if
    args and kwargs of the __init__ contain simple or MSONable elements.
    """

    def __init__(self, callable: Callable, set_atoms=False, *args, **kwargs):
        self.callable = callable
        self.module = callable.__module__
        self.name = callable.__name__
        self.set_atoms = set_atoms
        self.args = args
        self.kwargs = kwargs

    def generate(self, atoms=None, **kwargs):
        total_kwargs = dict(self.kwargs)
        total_kwargs.update(kwargs)
        if self.set_atoms:
            if not atoms:
                raise RuntimeError("atoms argument is required")
            if "atoms" not in total_kwargs:
                total_kwargs["atoms"] = atoms

        return self.callable(*self.args, **total_kwargs)

    def as_dict(self) -> dict:
        def recursive_as_dict(obj):
            if isinstance(obj, (list, tuple)):
                return [recursive_as_dict(it) for it in obj]
            if isinstance(obj, dict):
                return {kk: recursive_as_dict(vv) for kk, vv in obj.items()}
            if hasattr(obj, "as_dict"):
                return obj.as_dict()
            return obj

        d = {
            "set_atoms": self.set_atoms,
            "module": self.module,
            "name": self.name,
            "args": recursive_as_dict(self.args),
            "kwargs": recursive_as_dict(self.kwargs),
        }

        return d

    @classmethod
    def from_dict(cls, d: dict):
        module = d["module"]
        name = d["name"]
        mod = __import__(module, globals(), locals(), [name], 0)
        callable = getattr(mod, name)
        arg_decoded = MontyDecoder().process_decoded(d["args"])
        kwarg_decoded = MontyDecoder().process_decoded(d["kwargs"])
        set_atoms = d["set_atoms"]

        return cls(callable=callable, set_atoms=set_atoms, *arg_decoded, **kwarg_decoded)


def generate_optimizer(atoms: Atoms, optimizer: str, opt_kwargs):
    if not opt_kwargs:
        opt_kwargs = {}
    cls_ = getattr(ase.optimize, optimizer)
    return cls_(atoms, **opt_kwargs)
