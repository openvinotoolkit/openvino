# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import os

from importlib.metadata import PackageNotFoundError, version as _installed_version
from packaging.requirements import Requirement


def check_python_requirements(requirements_path: str) -> None:
    """
    Checks if the requirements defined in `requirements_path` are installed
    in the active Python environment, while also taking constraints.txt files
    into account.
    """

    constraints = {}
    constraints_path = None
    requirements = []

    # read requirements and find constraints file
    with open(requirements_path) as f:
        raw_requirements = f.readlines()
    for line in raw_requirements:
        if line.startswith("-c"):
            constraints_path = os.path.join(os.path.dirname(requirements_path), line.split(' ')[1][:-1])

    # read constraints if they exist
    if constraints_path:
        with open(constraints_path) as f:
            raw_constraints = f.readlines()
        for line in raw_constraints:
            if line.startswith("#") or line=="\n":
                continue
            line = line.replace("\n", "")
            # drop inline comments (e.g. "numpy>=1.16.6,<2.6.0  # frontends")
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            package, delimiter, constraint = re.split("(~|=|<|>|;)", line, maxsplit=1)
            if constraints.get(package) is None:
                constraints[package] = [delimiter + constraint]
            else:
                constraints[package].extend([delimiter + constraint])
        for line in raw_requirements:
            if line.startswith(("#", "-c")):
                continue
            line = line.replace("\n", "")
            if re.search("\W", line):
                requirements.append(line)
            else:
                constraint = constraints.get(line)
                if constraint:
                    for marker in constraint:
                        requirements.append(line+marker)
                else:
                    requirements.append(line)
    else:
        requirements = raw_requirements

    # Validate each requirement against the installed distributions. Uses
    # importlib.metadata + packaging instead of the removed pkg_resources API
    # (dropped in setuptools>=82), so the check keeps working with modern
    # setuptools. Raises to signal an unsatisfied requirement to the caller.
    for raw in requirements:
        raw = raw.strip()
        if not raw or raw.startswith(("#", "-")):
            continue
        req = Requirement(raw)
        # Skip requirements whose environment markers don't apply here.
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = _installed_version(req.name)
        except PackageNotFoundError:
            raise RuntimeError(f"Required package '{req.name}' is not installed")
        if req.specifier and installed not in req.specifier:
            raise RuntimeError(
                f"Installed '{req.name}=={installed}' does not satisfy requirement '{raw}'")
