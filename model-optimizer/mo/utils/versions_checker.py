"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import logging as log
import os
import re
import sys
from distutils.version import LooseVersion

modules = {
    "protobuf": "google.protobuf",
    "test-generator": "generator",
}
critical_modules = ["networkx"]

message = "\nDetected not satisfied dependencies:\n" \
          "{}\n" \
          "Please install required versions of components or use install_prerequisites script\n" \
          "{}\n" \
          "Note that install_prerequisites scripts may install additional components."


def check_python_version():
    """
    Checks python version to be greater or equal than 3.4
    :return: exit code (1 - error, None - successful)
    """
    if sys.version_info < (3, 4):
        print('Python version should be of version 3.4 or newer')
        return 1


def parse_versions_list(required_fw_versions: str, version_list: list()):
    """
    Parsing requirements versions
    :param required_fw_versions: String with fw versions from requirements file
    :param version_list: List for append
    :return: list of tuples of strings like (name_of_module, sign, version)

    Returned object is:
    [('tensorflow', '>=', '1.2.0'), ('networkx', '==', '2.1'), ('numpy', None, None)]
    """

    line = required_fw_versions.strip('\n')
    line = line.strip(' ')
    if line == '':
        return []
    splited_versions_by_conditions = re.split(r"==|>=|<=|>|<", line)
    splited_versions_by_conditions = [l.strip(',') for l in splited_versions_by_conditions]

    if len(splited_versions_by_conditions) == 0:
        return []
    if len(splited_versions_by_conditions) == 1:
        version_list.append((splited_versions_by_conditions[0], None, None))
    else:
        splited_required_versions= re.split(r",", line)
        for i, l in enumerate(splited_required_versions):
            comparisons = ['==', '>=', '<=', '<', '>']
            for comparison in comparisons:
                if comparison in l:
                    version_list.append((splited_versions_by_conditions[0], comparison, splited_versions_by_conditions[i + 1]))
                    break
    return version_list


def get_module_version_list_from_file(file_name: str):
    """
    Please do not add parameter type annotations (param:type).
    Because we import this file while checking Python version.
    Python 2.x will fail with no clear message on type annotations.

    Reads file with requirements
    :param file_name: Name of the requirements file
    :return: list of tuples of strings like (name_of_module, sign, version)

    File content example:
    tensorflow>=1.2.0
    networkx==2.1
    numpy

    Returned object is:
    [('tensorflow', '>=', '1.2.0'), ('networkx', '==', '2.1'), ('numpy', None, None)]
    """
    req_dict = list()
    with open(file_name) as f:
        for line in f:
            req_dict = parse_versions_list(line, req_dict)
    return req_dict


def version_check(name, installed_v, required_v, sign, not_satisfied_v, exit_code):
    """
    Please do not add parameter type annotations (param:type).
    Because we import this file while checking Python version.
    Python 2.x will fail with no clear message on type annotations.

    Evaluates comparison of installed and required versions according to requirements file of one module.
    If installed version does not satisfy requirements appends this module to not_stisfied_v list.
    :param name: module name
    :param installed_v: installed version of module
    :param required_v: required version of module
    :param sign: sing for comparison of required and installed versions
    :param not_satisfied_v: list of modules with not satisfying versions
    :param exit_code: flag of successful execution (0 - successful, 1 - error)
    :return: exit code
    """
    if sign is not None:
        req_ver = LooseVersion(required_v)
        satisfied = False
        if sign == '>':
            satisfied = installed_v > req_ver
        elif sign == '>=':
            satisfied = installed_v >= req_ver
        elif sign == '<=':
            satisfied = installed_v <= req_ver
        elif sign == '<':
            satisfied = installed_v < req_ver
        elif sign == '==':
            satisfied = installed_v == req_ver
        else:
            log.error("Error during version comparison")
    else:
        satisfied = True
    if not satisfied:
        not_satisfied_v.append((name, 'installed: {}'.format(installed_v), 'required: {}'.format(required_v)))
        if name in critical_modules:
            exit_code = 1
    return exit_code


def check_requirements(framework=None):
    """
    Please do not add parameter type annotations (param:type).
    Because we import this file while checking Python version.
    Python 2.x will fail with no clear message on type annotations.

    Checks if installed modules versions satisfy required versions in requirements file
    Logs a warning in case of permissible dissatisfaction
    Logs an error in cases of critical dissatisfaction
    :param framework: framework name
    :return: exit code (0 - execution successful, 1 - error)
    """
    if framework is None:
        framework_suffix = ""
    else:
        framework_suffix = "_{}".format(framework)
    file_name = "requirements{}.txt".format(framework_suffix)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, file_name))
    requirements_list = get_module_version_list_from_file(requirements_file)
    not_satisfied_versions = []
    exit_code = 0
    for name, key, required_version in requirements_list:
        try:
            importable_name = modules.get(name, name)
            exec("import {}".format(importable_name))
            installed_version = sys.modules[importable_name].__version__
            exit_code = version_check(name, installed_version, required_version, key, not_satisfied_versions, exit_code)
            exec("del {}".format(importable_name))
        except (AttributeError, ImportError):
            not_satisfied_versions.append((name, 'not installed', 'required: {}'.format(required_version)))
            exit_code = 1
            continue
        except Exception as e:
            log.error('Error happened while importing {} module. It may happen due to unsatisfied requirements of '
                      'that module. Please run requirements installation script once more.\n'
                      'Details on module importing failure: {}'.format(name, e))
            not_satisfied_versions.append((name, 'package error', 'required: {}'.format(required_version)))
            exit_code = 1
            continue

    if len(not_satisfied_versions) != 0:
        extension = 'bat' if os.name == 'nt' else 'sh'
        install_file = 'install_prerequisites{0}.{1}'.format(framework_suffix, extension)
        helper_command = os.path.join(os.path.dirname(requirements_file), 'install_prerequisites', install_file)
        missed_modules_message = ""
        for module in not_satisfied_versions:
            missed_modules_message += "\t{}: {}, {}\n".format(module[0], module[1], module[2])
        if exit_code:
            log.error(message.format(missed_modules_message, helper_command))
        else:
            log.error(message.format(missed_modules_message, helper_command), extra={'is_warning': True})
    return exit_code
