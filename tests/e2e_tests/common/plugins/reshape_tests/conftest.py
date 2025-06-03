import inspect
import itertools
import logging as log
import os
import sys
import traceback
from contextlib import contextmanager
from copy import copy, deepcopy
from types import SimpleNamespace

import pytest
import yaml

import e2e_tests.common.plugins.common.base_conftest as base
from e2e_tests.test_utils.reshape_tests_utils import should_run_reshape, get_reshape_configurations, \
    get_reshape_pipeline_pairs, batch_was_changed
from e2e_tests.test_utils.test_utils import class_factory, BrokenTest, BrokenTestException
from e2e_tests.common.env_utils import fix_env_conf
from e2e_tests.common.plugins.e2e_test.conftest import satisfies_all_rules, unwrap_rules
from e2e_tests.test_utils.env_tools import Environment


@contextmanager
def import_from(path):
    """ Set import preference to path"""
    os.sys.path.insert(0, os.path.realpath(path))
    yield
    os.sys.path.remove(os.path.realpath(path))


def set_env_for_reshape(metafunc):
    """Setup test environment."""
    with open(metafunc.config.getoption('env_conf'), "r") as env_conf:
        Environment.env = fix_env_conf(yaml.load(env_conf, Loader=yaml.FullLoader),
                                       root_path=str(metafunc.config.rootdir))

    with open(metafunc.config.getoption('test_conf'), "r") as test_conf:
        Environment.tconf = yaml.load(test_conf, Loader=yaml.FullLoader)

    with open(metafunc.config.getoption('reshape_rules_conf'), "r") as reshape_rules_conf:
        Environment.reshape_rules = unwrap_rules(yaml.load(reshape_rules_conf, Loader=yaml.FullLoader))

    with open(metafunc.config.getoption('dynamism_rules_conf'), "r") as dynamism_rules_conf:
        Environment.dynamism_rules = unwrap_rules(yaml.load(dynamism_rules_conf, Loader=yaml.FullLoader))


def read_reshape_test_config(required_args, test_config, reshape_rules_config=None):
    """Read test configuration file and return cartesian product of found
    parameters (filtered and full).
    """

    def prepare_test_params(keys, values):
        params = []
        parameters = itertools.product(*values)
        for parameter_set in parameters:
            named_params = dict(zip(keys, parameter_set))
            if satisfies_all_rules(named_params, reshape_rules_config):
                params.append(named_params)
        return params

    # sort dictionary items to enforce same order in different python runs
    keys = list(test_config.keys())
    vals = list(test_config.values())
    required_args_ind = [i for i, key in enumerate(keys) if key in required_args]
    req_keys = [keys[i] for i in required_args_ind]
    req_vals = [vals[i] for i in required_args_ind]
    req_params = prepare_test_params(req_keys, req_vals)

    addit_args_ind = set(range(len(keys))) - set(required_args_ind)
    addit_keys = [keys[i] for i in addit_args_ind]
    addit_vals = [vals[i] for i in addit_args_ind]
    addit_args = dict(zip(addit_keys, addit_vals))

    return req_params, addit_args


def pytest_generate_tests(metafunc):
    """Pytest hook for test generation.

    Generate parameterized tests from discovered modules and test config
    parameters.
    """
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)
    set_env_for_reshape(metafunc)
    reshape_test_classes, broken_modules = base.find_tests(metafunc.config.getoption('modules'),
                                                           attributes=['__is_test_config__'])
    for module in broken_modules:
        log.error("Broken module: {}. Import failed with error: {}".format(module[0], module[1]))

    reshape_test_cases = []
    reshape_test_ids = []
    reshape_configurations_list = []
    dynamism_type = metafunc.config.getoption('dynamism_type')
    consecutive_infer = metafunc.config.getoption('consecutive_infer')
    skip_ir_generation = metafunc.config.getoption('skip_ir_generation')

    # batch was set explicitly because reshape and dynamism tests do not use this parameter,
    # but it is required in e2e
    if len(Environment.tconf['batch']) > 1 or Environment.tconf['batch'][0] != 1:
        Environment.tconf['batch'] = [1]
        log.warning("batch was set explicitly to '1' because reshape and dynamism tests do not use this parameter,"
                    " but it is required to be in e2e")

    for reshape_test in reshape_test_classes:
        required_args = [arg for arg in inspect.signature(reshape_test.__init__).parameters.keys()]
        required_args.extend(getattr(metafunc, 'test_add_args_to_parametrize', []))
        rules = Environment.dynamism_rules if dynamism_type == "negative_ones" or dynamism_type == "range_values" \
            else Environment.reshape_rules

        params_list, addit_params_dict = read_reshape_test_config(required_args, Environment.tconf, rules)

        for _params in params_list:
            params = copy(_params)

            name = reshape_test.__name__
            test_id = "{}_{}".format(name, "_".join(
                "{}_{}".format(key, val) for (key, val) in sorted(params.items()) if key not in ['batch']))
            params.update({"skip_ir_generation": skip_ir_generation})
            try:
                reshape_test_case = reshape_test(**params, **addit_params_dict, **{"required_params": params},
                                                 test_id=test_id)
                if not should_run_reshape(reshape_test_case):
                    break
                configurations = get_reshape_configurations(reshape_test_case, dynamism_type)

            except Exception as e:
                configurations = [SimpleNamespace(shapes={}, changed_dims={}, layout={}, default_shapes={})]
                tb = traceback.format_exc()
                broken_test = class_factory(cls_name=name, cls_kwargs={"__name__": name, **params, **addit_params_dict,
                                                                       "required_params": params}, BaseClass=BrokenTest)
                reshape_test_case = broken_test(test_id=test_id, exception=e,
                                                fail_message="Test {} is broken and fails "
                                                             "with traceback {}".format(name, tb))

            if not getattr(reshape_test_case, "__do_not_run__", False):
                if configurations:
                    for configuration in configurations:
                        configuration.skip_ir_generation = skip_ir_generation
                        params_for_satisfaction = {"model": name, **params}

                        if dynamism_type == "negative_ones" or dynamism_type == "range_values":
                            reshape_test_id = test_id + "_".join(
                                "_{}_{}".format(k, v[0:]) for (k, v) in configuration.shapes.items())
                            if satisfies_all_rules(params_for_satisfaction, rules, can_partially_match=False):
                                reshape_test_ids.append(reshape_test_id)
                                reshape_test_cases.append(reshape_test_case)
                                configuration.consecutive_infer = consecutive_infer
                                configuration.dynamism_type = dynamism_type
                                reshape_configurations_list.append(configuration)
                        else:
                            requested_reshape_pairs = get_reshape_pipeline_pairs(reshape_test_case)
                            for reshape_pair in requested_reshape_pairs:
                                configuration = deepcopy(configuration)
                                reshape_test_id = test_id + "_".join(
                                    "_{}_{}".format(k, v[0:]) for (k, v) in configuration.shapes.items())
                                # check if there a point to run IE_SBS pipeline
                                if 'IE_SBS' in reshape_pair:
                                    batch = batch_was_changed(configuration.shapes, configuration.changed_dims,
                                                              configuration.layout, configuration.default_shapes)
                                    if not batch:
                                        continue

                                configuration.reshape_pair = reshape_pair
                                reshape_test_id = reshape_test_id + "{}".format(reshape_pair)
                                if satisfies_all_rules(params_for_satisfaction, rules, can_partially_match=False):
                                    reshape_test_ids.append(reshape_test_id)
                                    reshape_test_cases.append(reshape_test_case)
                                    reshape_configurations_list.append(configuration)
                                else:
                                    pass

    if reshape_test_cases:
        pairs_of_shape_and_test_case = list(zip(reshape_test_cases, reshape_configurations_list))
        metafunc.parametrize(argnames='instance,configuration',
                             argvalues=pairs_of_shape_and_test_case,
                             ids=reshape_test_ids)


def pytest_collection_modifyitems(items):
    """ Pytest hook for items collection. """

    for i in list(items):
        if not hasattr(i, 'callspec'):
            items.remove(i)

    items.sort(key=lambda item: (item.callspec.params['instance'].batch,
                                 item.callspec.params['instance'].__class__.__name__))

    pytorch_original_tests = []
    for i in items:
        test_name = i.name.replace(i.originalname, '').replace('[', '').lower()
        pytorch_original_tests.append(test_name.startswith('pytorch'))

    # this WA required because of: 1. pytorch leaks 2. e2e lack of possibility to put every test in multiprocessing
    # on Win and MacOS
    pytorch_group_marked = 0
    # if number inside the range will be changed there should be according changes in pytest.ini file
    group_names = [f'Pytorch_group_{j}' for j in range(7)]
    bucket_size = sum(pytorch_original_tests) // len(group_names)
    current_group_idx = 0

    for num, test in enumerate(items):
        instance = test.callspec.params['instance']
        target_test_runner = test.originalname

        try:
            if pytorch_original_tests[num]:
                test.add_marker(group_names[current_group_idx])
                pytorch_group_marked += 1
                if pytorch_group_marked % bucket_size == 0 and pytorch_group_marked < bucket_size * len(group_names):
                    current_group_idx += 1

            base.set_pytest_marks(_test=test, _object=instance, _runner=target_test_runner, log=log)
        except BrokenTestException as e:
            test.add_marker("broken_test")
            continue
