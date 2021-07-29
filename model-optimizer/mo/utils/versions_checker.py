# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
import re
import sys
from distutils.version import LooseVersion

modules = {
    "protobuf": "google.protobuf",
    "test-generator": "generator",
}
critical_modules = ["networkx", "defusedxml", "numpy"]

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


def parse_and_filter_versions_list(required_fw_versions, version_list, env_setup):
    """
    Please do not add parameter type annotations (param:type).
    Because we import this file while checking Python version.
    Python 2.x will fail with no clear message on type annotations.

    Parsing requirements versions for a dependency and filtering out requirements that
    satisfy environment setup such as python version.
    if environment version (python_version, etc.) is satisfied
    :param required_fw_versions: String with fw versions from requirements file
    :param version_list: List for append
    :param env_setup: a dictionary with environment setup
    :return: list of tuples of strings like (name_of_module, sign, version)

    Examples of required_fw_versions:
    'tensorflow>=1.15.2,<2.0; python_version < "3.8"'
    'tensorflow>=2.0'

    Returned object is:
    [('tensorflow', '>=', '1.2.0'), ('networkx', '==', '2.1'), ('numpy', None, None)]
    """

    line = required_fw_versions.strip('\n')
    line = line.strip(' ')
    if line == '':
        return version_list
    split_requirement = line.split(";")

    # check environment marker
    if len(split_requirement) > 1:
        env_req = split_requirement[1]
        if any([x in split_requirement[1] for x in [' and ', ' or ']]):
            log.error("The version checker doesn't support environment marker combination and it will be ignored: {}"
                      "".format(split_requirement[1]), extra={'is_warning': True})
            return version_list
        split_env_req = re.split(r"==|>=|<=|>|<|~=|!=", env_req)
        split_env_req = [l.strip(',') for l in split_env_req]
        env_marker = split_env_req[0].strip(' ')
        if env_marker == 'python_version' and env_marker in env_setup:
            installed_python_version = env_setup['python_version']
            env_req_version_list = []
            split_required_versions = re.split(r",", env_req)
            for i, l in enumerate(split_required_versions):
                for comparison in ['==', '>=', '<=', '<', '>', '~=']:
                    if comparison in l:
                        required_version = split_env_req[i + 1].strip(' ').replace("'", "").replace('"', '')
                        env_req_version_list.append((env_marker, comparison, required_version))
                        break
            not_satisfied_list = []
            for name, key, required_version in env_req_version_list:
                version_check(name, installed_python_version, required_version,
                              key, not_satisfied_list)
            if len(not_satisfied_list) > 0:
                # this python_version requirement is not satisfied to required environment
                # and requirement for a dependency will be skipped
                return version_list
        elif env_marker == 'sys_platform' and env_marker in env_setup:
            split_env_req[1] = split_env_req[1].strip(' ').replace("'", "").replace('"', '')
            if '==' in env_req:
                if env_setup['sys_platform'] != split_env_req[1]:
                    # this sys_platform requirement is not satisfied to required environment
                    # and requirement for a dependency will be skipped
                    return version_list
            elif '!=' in env_req:
                if env_setup['sys_platform'] == split_env_req[1]:
                    # this sys_platform requirement is not satisfied to required environment
                    # and requirement for a dependency will be skipped
                    return version_list
            else:
                log.error("Error during platform version check, line: {}".format(line))
        else:
            log.error("{} is unsupported environment marker and it will be ignored".format(env_marker),
                      extra={'is_warning': True})

    # parse a requirement for a dependency
    requirement = split_requirement[0]
    split_versions_by_conditions = re.split(r"==|>=|<=|>|<|~=", requirement)
    split_versions_by_conditions = [l.strip(',').strip(' ') for l in split_versions_by_conditions]

    if len(split_versions_by_conditions) == 0:
        return version_list
    if len(split_versions_by_conditions) == 1:
        version_list.append((split_versions_by_conditions[0], None, None))
    else:
        split_required_versions= re.split(r",", requirement)
        for i, l in enumerate(split_required_versions):
            for comparison in ['==', '>=', '<=', '<', '>', '~=']:
                if comparison in l:
                    version_list.append((split_versions_by_conditions[0], comparison, split_versions_by_conditions[i + 1]))
                    break
    return version_list


def get_module_version_list_from_file(file_name, env_setup):
    """
    Please do not add parameter type annotations (param:type).
    Because we import this file while checking Python version.
    Python 2.x will fail with no clear message on type annotations.

    Reads file with requirements
    :param file_name: Name of the requirements file
    :param env_setup: a dictionary with environment setup elements
    :return: list of tuples of strings like (name_of_module, sign, version)

    File content example:
    tensorflow>=1.2.0
    networkx==2.1
    numpy

    Returned object is:
    [('tensorflow', '>=', '1.2.0'), ('networkx', '==', '2.1'), ('numpy', None, None)]
    """
    req_dict = []
    with open(file_name) as f:
        for line in f:
            # handle comments
            line = line.split('#')[0]

            req_dict = parse_and_filter_versions_list(line, req_dict, env_setup)
    return req_dict


def version_check(name, installed_v, required_v, sign, not_satisfied_v):
    """
    Please do not add parameter type annotations (param:type).
    Because we import this file while checking Python version.
    Python 2.x will fail with no clear message on type annotations.

    Evaluates comparison of installed and required versions according to requirements file of one module.
    If installed version does not satisfy requirements appends this module to not_satisfied_v list.
    :param name: module name
    :param installed_v: installed version of module
    :param required_v: required version of module
    :param sign: sing for comparison of required and installed versions
    :param not_satisfied_v: list of modules with not satisfying versions
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
        elif sign == '~=':
            req_ver_list = req_ver.vstring.split('.')
            if 'post' in req_ver_list[-1]:
                assert len(req_ver_list) >= 3, 'Error during {} module version checking: {} {} {}, please check ' \
                                               'required version of this module in requirements_*.txt file!'\
                    .format(name, installed_v, sign, required_v)
                req_ver_list.pop(-1)
            idx = len(req_ver_list) - 1
            satisfied = installed_v >= req_ver and (installed_v.split('.')[:idx] == req_ver_list[:idx])
        else:
            log.error("Error during version comparison")
    else:
        satisfied = True
    if not satisfied:
        not_satisfied_v.append((name, 'installed: {}'.format(installed_v), 'required: {} {}'.format(sign, required_v)))


def get_environment_setup(framework):
    """
    Get environment setup such as Python version, TensorFlow version
    :param framework: framework name
    :return: a dictionary of environment variables
    """
    env_setup = dict()
    python_version = "{}.{}.{}".format(sys.version_info.major,
                                       sys.version_info.minor,
                                       sys.version_info.micro)
    env_setup['python_version'] = python_version
    try:
        if framework == 'tf':
            exec("import tensorflow")
            env_setup['tensorflow'] = sys.modules["tensorflow"].__version__
            exec("del tensorflow")
    except (AttributeError, ImportError):
        pass
    env_setup['sys_platform'] = sys.platform
    return env_setup


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
    env_setup = get_environment_setup(framework)
    if framework is None:
        framework_suffix = ""
    elif framework == "tf":
        if "tensorflow" in env_setup and env_setup["tensorflow"] >= LooseVersion("2.0.0"):
            framework_suffix = "_tf2"
        else:
            framework_suffix = "_tf"
    else:
        framework_suffix = "_{}".format(framework)

    file_name = "requirements{}.txt".format(framework_suffix)
    requirements_file = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, file_name))
    requirements_list = get_module_version_list_from_file(requirements_file, env_setup)
    not_satisfied_versions = []
    exit_code = 0
    for name, key, required_version in requirements_list:
        try:
            importable_name = modules.get(name, name)
            exec("import {}".format(importable_name))
            installed_version = sys.modules[importable_name].__version__
            version_check(name, installed_version, required_version, key, not_satisfied_versions)
            exec("del {}".format(importable_name))
        except (AttributeError, ImportError):
            # we need to raise error only in cases when import of critical modules is failed
            if name in critical_modules:
                exit_code = 1
            if key is not None and required_version is not None:
                not_satisfied_versions.append((name, 'not installed', 'required: {} {}'.format(key, required_version)))
            else:
                not_satisfied_versions.append((name, 'not installed', ''))
            continue
        except Exception as e:
            log.error('Error happened while importing {} module. It may happen due to unsatisfied requirements of '
                      'that module. Please run requirements installation script once more.\n'
                      'Details on module importing failure: {}'.format(name, e))
            not_satisfied_versions.append((name, 'package error', 'required: {} {}'.format(key, required_version)))
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
