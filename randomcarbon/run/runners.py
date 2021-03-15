from typing import List, Union
import multiprocessing
import queue
import time
import logging
import random
import traceback
import os
from abc import ABC, abstractmethod
from collections import deque
from randomcarbon.run.ase import relax
from randomcarbon.run.generator import Generator
from randomcarbon.evolution.core import Evolver, Filter, Blocker, evolve_structure
from randomcarbon.utils.structure import set_properties, get_properties, to_primitive
from randomcarbon.utils.structure import to_supercell, set_structure_id, get_property
from randomcarbon.utils.factory import Factory
from randomcarbon.output.store import Store
from randomcarbon.output.results import store_results
from randomcarbon.output.taggers.core import Tagger
from pymatgen.core.structure import Structure
from monty.serialization import dumpfn
from monty.os import makedirs_p

logger = logging.getLogger(__name__)


class BaseRunner(ABC):
    """
    Base runner. Defines the basic __init__ and provides a common place for the
    portion of code that executes the calculation and evolution.
    """

    def __init__(self, calculator_factory: Factory, evolvers: List[Union[Evolver, List]],
                 blockers: List[Blocker] = None, filters: List[Filter] = None,
                 fmax: float = 0.05, steps: int = 1000, constraints: List[Factory] = None,
                 optimizer: str = "BFGS", opt_kwargs: dict = None,
                 allow_not_converged: bool = False, store: Store = None,
                 spacegroup_primitive: int = None, taggers: List[Tagger] = None):

        self.calculator_factory = calculator_factory
        self.evolvers = evolvers
        self.blockers = blockers
        self.filters = filters
        self.fmax = fmax
        self.steps = steps
        self.constraints = constraints
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        self.allow_not_converged = allow_not_converged
        self.store = store
        self.spacegroup_primitive = spacegroup_primitive
        self.taggers = taggers
        self.counter = 0

    def _execute(self, structure: Structure) -> List[Structure]:
        # set the id before running to be sure to have it at any time.
        # Will not overwrite an existing one.
        set_structure_id(structure)
        run_structure = structure
        try:
            convert = False
            #TODO check if it is fine not to allow to customize symprec here
            if self.spacegroup_primitive:
                try:
                    run_structure, convert, conv_matrix = to_primitive(
                        structure=structure, spacegroup=self.spacegroup_primitive, preserve_properties=True)
                    logger.debug(f"Running structure with {len(run_structure)} instead of {len(structure)}")
                except:
                    # in case of errors in the conversion, just run the original one.
                    # it seems in can happen with spglib failing to provide the primitive.
                    pass

            relaxed_structure = relax(structure=run_structure, calculator=self.calculator_factory.generate(),
                                      fmax=self.fmax, steps=self.steps, constraints=self.constraints,
                                      optimizer=self.optimizer, opt_kwargs=self.opt_kwargs,
                                      allow_not_converged=self.allow_not_converged, set_energy_in_structure=True,
                                      preserve_properties=True)

            # if the structure is not relaxed it will be None. Stop here the evolution.
            if not relaxed_structure:
                return []

            logger.info(f"structure relaxed")

            if convert and self.spacegroup_primitive:
                properties = get_properties(relaxed_structure)
                tmp_relaxed_structure = to_supercell(
                    structure=relaxed_structure, conversion_matrix=conv_matrix, preserve_properties=True
                )

                if not tmp_relaxed_structure:
                    structure_id = properties.get('structure_id')
                    logger.warning(f"Structure {structure_id} was not converted back from the primitive."
                                   f"Halting and dumping to json file. History: {properties.get('history')}")
                    dumpfn(f"{structure_id}.json.gz", relaxed_structure)
                    return []

                properties["energy"] *= len(structure) / len(relaxed_structure)
                relaxed_structure = tmp_relaxed_structure
                set_properties(relaxed_structure, properties)

            new_structures = evolve_structure(relaxed_structure, evolvers=self.evolvers,
                                              blockers=self.blockers, filters=self.filters)

            store_results(relaxed_structure, store=self.store, taggers=self.taggers)

            logger.info(f"new structures generated: {len(new_structures)}")

            return new_structures

        except:
            # in case of error do not raise but just return an empty list. This will allow
            # the evolution of other structures to continue when running with parallel runners.
            structure_id = get_property(structure, "structure_id")
            logger.warning(f"An error occurred when running structure {structure_id}: \n" + traceback.format_exc())
            makedirs_p("run_errors")
            dumpfn({"structure": run_structure, "history": get_property(run_structure, "history"),
                    "error": traceback.format_exc()},
                   os.path.join("run_errors", f"{structure_id}_error.json.gz"))
            return []


class SequentialRunner(BaseRunner):
    """
    Base class for the runners that are executed sequentially with a single initial structure.
    """

    def __init__(self, calculator_factory: Factory, evolvers: List[Union[Evolver, List]],
                 initial_structure: Structure, blockers: List[Blocker] = None, filters: List[Filter] = None,
                 fmax: float = 0.05, steps: int = 1000, constraints: List[Factory] = None,
                 optimizer: str = "BFGS", opt_kwargs: dict = None,
                 allow_not_converged: bool = False, store: Store = None,
                 spacegroup_primitive: int = None, taggers: List[Tagger] = None):

        super().__init__(calculator_factory=calculator_factory, evolvers=evolvers,
                         blockers=blockers, filters=filters,
                         fmax=fmax, steps=steps, constraints=constraints,
                         optimizer=optimizer, opt_kwargs=opt_kwargs,
                         allow_not_converged=allow_not_converged, store=store,
                         spacegroup_primitive=spacegroup_primitive, taggers=taggers)

        self.initial_structure = initial_structure
        set_properties(self.initial_structure, {"history": []})

    @abstractmethod
    def run(self, max_structures: int = None):
        pass


class BranchingRunner(SequentialRunner):
    """
    A runner that works sequentially on a queue of structures. The queue can be increased at
    each step if new structures are generated. This allows to follow different branching paths
    in the generation, starting from a single initial structure.
    """

    def run(self, max_structures: int = None):
        queue = deque()
        queue.append(self.initial_structure)

        self.store.connect()

        while (max_structures is None or self.counter < max_structures) and len(queue) > 0:
            structure = queue.popleft()

            new_structures = self._execute(structure)

            queue.extend(new_structures)
            self.counter += 1

        if max_structures and self.counter >= max_structures:
            logger.info(f"Stopping due to maximum number of structures reached: {self.counter}")

        self.store.close()


class SerialRunner(SequentialRunner):
    """
    A simple runner that starting from a structure will at each step pick a single new
    structure among those generated and continue with just that single one.
    """

    def run(self, max_structures: int = None):

        next_structure = self.initial_structure

        self.store.connect()

        while (max_structures is None or self.counter < max_structures) and next_structure:
            new_structures = self._execute(next_structure)

            if new_structures:
                next_structure = new_structures[0]
            else:
                next_structure = None

            self.counter += 1

        if max_structures and self.counter >= max_structures:
            logger.info(f"Stopping due to maximum number of structures reached: {self.counter}")

        self.store.close()


RUNNERS_NAME = {"SerialRunner": SerialRunner,
                "BranchingRunner": BranchingRunner}


class MultiStructureRunner(BaseRunner):
    """
    A Runner that repeatedly runs structures, if the list of evolved structure is emptied a
    new initial structure will be generated. Relies on the SequentialRunners for the
    internal execution.
    TODO if this kind of runner is useful replace the Evolver in input as generator with
    a more suitable object.
    """
    def __init__(self, calculator_factory: Factory, evolvers: List[Union[Evolver, List]],
                 generator: Generator, inner_runner: str = "SerialRunner",
                 blockers: List[Blocker] = None, filters: List[Filter] = None, fmax: float = 0.05, steps: int = 1000,
                 constraints: List[Factory] = None, optimizer: str = "BFGS", opt_kwargs: dict = None,
                 allow_not_converged: bool = False, store: Store = None,
                 spacegroup_primitive: int = None, taggers: List[Tagger] = None):
        super().__init__(calculator_factory=calculator_factory, evolvers=evolvers,
                         blockers=blockers, filters=filters,
                         fmax=fmax, steps=steps, constraints=constraints,
                         optimizer=optimizer, opt_kwargs=opt_kwargs,
                         allow_not_converged=allow_not_converged, store=store,
                         spacegroup_primitive=spacegroup_primitive, taggers=taggers)

        if inner_runner not in (RUNNERS_NAME.keys()):
            raise ValueError(f"Unknown runner {inner_runner}")
        self.inner_runner = inner_runner

        self.generator = generator

    def run(self, max_structures: int = None):
        while self.counter < max_structures:
            runner_cls = RUNNERS_NAME[self.inner_runner]
            generated_structures = self.generator.generate()
            if generated_structures:
                initial_structure = generated_structures[0]
            else:
                break
            r = runner_cls(calculator_factory=self.calculator_factory, evolvers=self.evolvers,
                           initial_structure=initial_structure, blockers=self.blockers, filters=self.filters,
                           fmax=self.fmax, steps=self.steps, constraints=self.constraints,
                           optimizer=self.optimizer, opt_kwargs=self.opt_kwargs,
                           allow_not_converged=self.allow_not_converged, store=self.store,
                           taggers=self.taggers)

            r.run(max_structures - self.counter)
            self.counter += r.counter


class ParallelRunner(BaseRunner):
    """
    A runner parallelized with multiprocessing to run simultaneously several SequentialRunners
    at the same time. Each of the internal runner will be executed independently from the others.
    The number of processes spawn is determined by the number of initial structures passed.
    """

    def __init__(self, calculator_factory: Factory, evolvers: List[Union[Evolver, List]],
                 initial_structures: List[Structure], inner_runner: str = "SerialRunner",
                 blockers: List[Blocker] = None, filters: List[Filter] = None, fmax: float = 0.05, steps: int = 1000,
                 constraints: List[Factory] = None, optimizer: str = "BFGS", opt_kwargs: dict = None,
                 allow_not_converged: bool = False, store: Store = None,
                 spacegroup_primitive: int = None, taggers: List[Tagger] = None):

        super().__init__(calculator_factory=calculator_factory, evolvers=evolvers,
                         blockers=blockers, filters=filters,
                         fmax=fmax, steps=steps, constraints=constraints,
                         optimizer=optimizer, opt_kwargs=opt_kwargs,
                         allow_not_converged=allow_not_converged, store=store,
                         spacegroup_primitive=spacegroup_primitive, taggers=taggers)

        self.initial_structures = initial_structures
        for s in initial_structures:
            set_properties(s, {"history": []})

        if inner_runner not in (RUNNERS_NAME.keys()):
            raise ValueError(f"Unknown runner {inner_runner}")
        self.inner_runner = inner_runner

    def run_process(self, initial_structure: Structure, max_structures: int = None):
        runner_cls = RUNNERS_NAME[self.inner_runner]
        r = runner_cls(calculator_factory=self.calculator_factory, evolvers=self.evolvers,
                       initial_structure=initial_structure, blockers=self.blockers, filters=self.filters,
                       fmax=self.fmax, steps=self.steps, constraints=self.constraints,
                       optimizer=self.optimizer, opt_kwargs=self.opt_kwargs,
                       allow_not_converged=self.allow_not_converged, store=self.store,
                       spacegroup_primitive=self.spacegroup_primitive)
        r.run(max_structures)

    def run(self, max_structures: int = None):
        procs = []
        for s in self.initial_structures:
            p = multiprocessing.Process(target=self.run_process, args=(s, max_structures))
            procs.append(p)
            p.start()

        for proc in procs:
            proc.join()


class BranchingParallelRunner(BaseRunner):
    """
    A runner that works in parallel on a queue of structures. The queue can be increased at
    each step if new structures are generated. All the processing work concurrently on the same
    queue. This allows to follow different branching paths in the generation, starting from
    a single initial structure.
    """

    def __init__(self, calculator_factory: Factory, evolvers: List[Union[Evolver, List]],
                 initial_structures: Union[Structure, List[Structure]],
                 blockers: List[Blocker] = None, filters: List[Filter] = None,
                 fmax: float = 0.05, steps: int = 1000, constraints: List[Factory] = None,
                 optimizer: str = "BFGS", opt_kwargs: dict = None,
                 allow_not_converged: bool = False, store: Store = None,
                 spacegroup_primitive: int = None, taggers: List[Tagger] = None,
                 generators: Union[Generator, List[Generator]] = None):

        super().__init__(calculator_factory=calculator_factory, evolvers=evolvers,
                         blockers=blockers, filters=filters,
                         fmax=fmax, steps=steps, constraints=constraints,
                         optimizer=optimizer, opt_kwargs=opt_kwargs,
                         allow_not_converged=allow_not_converged, store=store,
                         spacegroup_primitive=spacegroup_primitive, taggers=taggers)

        if not isinstance(initial_structures, (list, tuple)):
            initial_structures = [initial_structures]
        self.initial_structures = initial_structures
        self.counter = multiprocessing.Value("i", 0)
        self.queue = multiprocessing.Queue()
        self.manager = multiprocessing.Manager()
        self.process_status = self.manager.dict()
        for structure in self.initial_structures:
            self.queue.put(structure)
            set_properties(structure, {"history": []})
        if not generators:
            generators = []
        elif not isinstance(generators, (list, tuple)):
            generators = [generators]
        self.generators = generators

    def run_process(self, max_structures: int, lock: multiprocessing.Lock):

        pid = multiprocessing.current_process().pid
        self.process_status[pid] = True

        self.store.connect()

        # After checking the counter, it checks if any of the process of the processes
        # is running something. The lock is basically for operating on the dictionary
        # while avoiding race conditions. The sleep is done outside the lock to avoid
        # starving of some processes that could otherwise never get to set their
        # process to False.
        try:
            while max_structures is None or self.counter.value < max_structures:
                try:
                    with lock:
                        if not any(self.process_status.values()):
                            # death pill
                            break
                        self.process_status[pid] = False
                        structure = self.queue.get_nowait()
                        self.process_status[pid] = True
                except queue.Empty:
                    if self.generators:
                        generator = random.choice(self.generators)
                        generated_structures = generator.generate()
                        if not generated_structures:
                            time.sleep(3)
                            continue
                        structure = generated_structures[0]
                        set_structure_id(structure)
                    else:
                        time.sleep(3)
                        continue
                logger.info(f"process {pid} starts running. Size of the queue: {self.queue.qsize()}")
                new_structures = self._execute(structure)

                for ns in new_structures:
                    self.queue.put(ns)

                # if no max_structures do not waste time with putting the lock and increasing the
                # shared counter
                if max_structures:
                    with self.counter.get_lock():
                        self.counter.value += 1
        finally:
            # if all the processes have stopped empty the queue
            with lock:
                self.process_status[pid] = False
                if not any(self.process_status.values()):
                    try:
                        while True:
                            # it seems that a get_nowait() sometimes results in this passing
                            # without emptying the queue. Not sure why this happens.
                            self.queue.get(timeout=0.5)
                    except queue.Empty:
                        pass
            logger.info(f"process {pid} is shutting down. Size of the queue: {self.queue.qsize()}")
            self.store.close()

    def run(self, nprocs: int, max_structures: int = None):

        lock = multiprocessing.Lock()

        procs = []
        for i in range(nprocs):
            p = multiprocessing.Process(target=self.run_process, args=(max_structures, lock))
            procs.append(p)
            p.start()

        for proc in procs:
            proc.join()
