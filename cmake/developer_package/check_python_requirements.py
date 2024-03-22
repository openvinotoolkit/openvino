import pkg_resources
import re
import os
import sys


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
    pkg_resources.require(requirements)


def check_python_requirements_pdm(toml_path: str):
    from pdm.environments import PythonEnvironment as Environment
    from pdm.project import Project
    from pdm.core import Core
    from pdm.cli.commands.venv.utils import get_venv_with_name

    def are_dependencies_installed(dependencies_to_check, installed_dependencies):
        for key in dependencies_to_check:
            print(key, key in installed_dependencies)
        return all(key in installed_dependencies for key in dependencies_to_check)

    directory = os.path.dirname(toml_path)

    core = Core()
    project = Project(core=core, root_path=directory)

    venv = get_venv_with_name(project, "for-build")
    environment = Environment(project, python=str(venv.interpreter))

    dependencies_to_check = project.get_dependencies(group="build").keys()
    installed_packages = environment.get_working_set() #returns ChainMap

    # print(environment.get_working_set())
    return sys.exit(0 if are_dependencies_installed(dependencies_to_check, installed_packages) else 1)
