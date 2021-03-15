import os
import tempfile
import contextlib
import subprocess
import logging
from typing import Optional, Dict
from monty.os import cd
from randomcarbon.rings.input import RingsInput, RingMethod
from randomcarbon.rings.output import RingsList

logger = logging.getLogger(__name__)


def run_rings(rings_input: RingsInput, workdir: str = None, executable: str = "rings",
              irreducible: bool = True) -> Optional[Dict[RingMethod, RingsList]]:
    """
    Executes the rings code based on the input.

    Args:
        rings_input: an input for the rings code.
        workdir: the directory were the calculation is executed. If None a new temporary directory
            will be created and deleted at the end of the calculation.
        executable: the full path to the rings executable if not in the PATH.
        irreducible: if the irreducible rings should be parsed from the outputs.

    Returns:
        a dictionary with the extracted RingsList for all the required methods.
    """
    if not workdir:
        cm = tempfile.TemporaryDirectory()
    else:
        cm = contextlib.nullcontext()

    with cm as tmp_dir:
        run_dir = workdir or tmp_dir
        with cd(run_dir):
            input_filename = "rings_input"
            output_filename = "rings_output"
            rings_input.write(run_dir, input_filename)
            command = [executable, input_filename]

            with open(output_filename, 'w') as stdout:
                process = subprocess.Popen(command, stdout=stdout)

            process.communicate()
            returncode = process.returncode

            failed = True
            if os.path.isfile(output_filename):
                with open(output_filename) as f:
                    out_string = f.read()
                failed = "All files have been succesfully written" not in out_string

            if returncode != 0 or failed:
                logger.warning("error while running rings")
                return None

            results = {}
            for method in rings_input.methods:
                results[method.value] = RingsList.from_dir("rstat", method=method, structure=rings_input.structure,
                                                           irreducible=irreducible)

            return results
