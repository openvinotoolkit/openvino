"""
 Copyright (c) 2018 Intel Corporation

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

modules = {"protobuf": "google.protobuf"}
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


def get_module_version_list_from_file(file_name):
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
            line = line.strip('\n')
            line = line.strip(' ')
            if line == '':
                continue
            splited_line = re.split(r"==|>=|<=|>|<", line)
            if len(splited_line) == 1:
                req_dict.append((splited_line[0], None, None))
            else:
                if '==' in line:
                    req_dict.append((splited_line[0], '==', splited_line[1]))
                elif '>=' in line:
                    req_dict.append((splited_line[0], '>=', splited_line[1]))
                elif '<=' in line:
                    req_dict.append((splited_line[0], '<=', splited_line[1]))
                elif '<' in line:
                    req_dict.append((splited_line[0], '<', splited_line[1]))
                elif '>' in line:
                    req_dict.append((splited_line[0], '>', splited_line[1]))
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
        satisfied = eval('installed_v{}req_ver'.format(sign))
    else:
        satisfied = True
    if not satisfied:
        not_satisfied_v.append((name, 'installed: {}'.format(installed_v), 'required: {}'.format(required_v)))
        if name in critical_modules:
            exit_code = 1
    return exit_code


def check_requirements(framework = None):
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
            exec("import {}".format(modules[name] if name in modules else name))
            installed_version = eval("{}.__version__".format(modules[name] if name in modules else name))
            exit_code = version_check(name, installed_version, required_version, key, not_satisfied_versions, exit_code)
            exec("del {}".format(modules[name] if name in modules else name))
        except (AttributeError, ImportError):
            not_satisfied_versions.append((name, 'not installed', 'required: {}'.format(required_version)))
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
