# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local pytest plugin for tests execution."""

import inspect
import itertools
import logging as log
# pylint:disable=import-error
import re
import os
import sys
import traceback
from copy import copy
from pathlib import Path

import pytest
import yaml
from _pytest.runner import show_test_item, call_runtest_hook, check_interactive_exception

import e2e_tests.common.plugins.common.base_conftest as base

from e2e_tests.test_utils.path_utils import DirLockingHandler
from e2e_tests.test_utils.test_utils import class_factory, BrokenTest, BrokenTestException
from e2e_tests.common import hook_utils
from e2e_tests.common.env_utils import fix_env_conf
from e2e_tests.common.logger import get_logger
from e2e_tests.common.marks import MarkRunType, MarkGeneral
from e2e_tests.test_utils.env_tools import Environment

logger = get_logger(__name__)


def __to_list(value):
    """Wrap non-list value in list."""
    if isinstance(value, list):
        return value
    return [value]


def set_env(metafunc):
    """Setup test environment."""
    with open(metafunc.config.getoption('env_conf'), "r") as env_conf:
        Environment.env = fix_env_conf(yaml.load(env_conf, Loader=yaml.FullLoader),
                                       root_path=str(metafunc.config.rootdir))

    with open(metafunc.config.getoption('test_conf'), "r") as test_conf:
        Environment.tconf = yaml.load(test_conf, Loader=yaml.FullLoader)

    with open(metafunc.config.getoption('base_rules_conf'), "r") as base_rules_conf:
        Environment.base_rules = unwrap_rules(yaml.load(base_rules_conf, Loader=yaml.FullLoader))


def unwrap_rules(rules_config):
    """Unwrap all rule values in rules config into a cartesian product.

    Example: {device: GPU, precision: [FP32, FP16]} => [{device: GPU, precision:
    FP32}, {device: GPU, precision: FP16}]
    """
    if not rules_config:
        return []
    for i, rules_dict in enumerate(rules_config):
        unwrapped_rules = []
        for rule in rules_dict['rules']:
            keys = rule.keys()
            vals = []
            for value in rule.values():
                vals.append([v for v in __to_list(value)])
            for rule_set in itertools.product(*vals):
                unwrapped_rules.append(dict(zip(keys, rule_set)))
        rules_config[i]['rules'] = unwrapped_rules
    return rules_config


def satisfies_rules(parameter_set, rules, filters, can_partially_match=False):
    """Check whether parameter_set satisfies rules.

    If there are no rules for such parameter_set, parameter_set is considered
    satisfactory (satisfies_rules returns True).

    By default (can_partially_match is False), parameters are filtered if rule
    value exactly matches the parameter value (e.g. 'CPU' == 'CPU').

    If can_partially_match is True, rule value may be a substring of a parameter
    value (e.g. 'CP' is substring of 'CPU'). Partial matching is useful when
    multiple models with similar name must be filtered, for example: MobileNet
    and MobileNet_v2.
    """

    def equal(a, b):
        """
        Check if a equals b
        or a match the rule 'not b'
        """
        if str(a).startswith('not'): 
            return a.replace('not ', '') != b
        return a == b

    def substr(a, b):
        """Check if a is substring of b"""
        return a in b

    satisfies = True
    # filter rules by non-matchable attributes
    match = substr if can_partially_match else equal
    applicable_rules = rules
    for key in filters:
        applicable_rules = list(filter(lambda rule: match(rule[key], parameter_set[key]), applicable_rules))
    # if there are no rules left, consider parameter_set satisfactory
    if not applicable_rules:
        return True
    # check whether parameter_set satisfies rules
    rule_satisfactions = []
    for rule in applicable_rules:
        common_keys = (set(parameter_set.keys()) & set(rule.keys())) - set(filters)
        if not common_keys:
            continue
        # all parameters must match for current rule to be satisfied by
        # parameter_set
        rule_satisfactions.append(
            all(equal(rule[k], parameter_set[k]) for k in common_keys))
    # there must be at least one match (True value) to consider parameter_set
    # satisfactory
    return satisfies & any(rule_satisfactions)


def satisfies_all_rules(values_set, rules_config, can_partially_match=False):
    """Check whether values_set satisfies all rules in rules_config.

    This function calls satisfies_rules for each suitable pair of
    rules and filters in rules configuration file.
    """
    satisfies = True
    for rules_dict in rules_config:
        rules = __to_list(rules_dict['rules'])
        filters = __to_list(rules_dict['filter_by'])
        # if key doesn't exist in the values_set, consider rules/filters do
        # not apply to these values
        if any(key not in values_set for key in filters):
            continue
        satisfies &= satisfies_rules(values_set, rules, filters,
                                     can_partially_match)
    return satisfies


def read_test_config(required_args, test_config, rules_config=None):
    """Read test configuration file and return cartesian product of found
    parameters (filtered and full).
    """

    def prepare_test_params(keys, values):
        params = []
        parameters = itertools.product(*values)
        for parameter_set in parameters:
            named_params = dict(zip(keys, parameter_set))
            if satisfies_all_rules(named_params, rules_config):
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
    set_env(metafunc)
    modules = metafunc.config.getoption('modules')
    test_classes, broken_modules = base.find_tests(modules, attributes=['__is_test_config__'])
    for module in broken_modules:
        log.error("Broken module: {}. Import failed with error: {}".format(module[0], module[1]))

    test_cases = []
    test_ids = []
    cpu_throughput_mode = metafunc.config.getoption("cpu_throughput_mode")
    gpu_throughput_mode = metafunc.config.getoption("gpu_throughput_mode")
    skip_ir_generation = metafunc.config.getoption("skip_ir_generation")

    for test in test_classes:
        setattr(test, 'convert_pytorch_to_onnx', metafunc.config.getoption('convert_pytorch_to_onnx'))
        required_args = list(inspect.signature(test.__init__).parameters.keys())[1:]
        required_args.extend(getattr(metafunc, 'test_add_args_to_parametrize', []))
        params_list, addit_params_dict = read_test_config(required_args, Environment.tconf, Environment.base_rules)
        for _params in params_list:
            params = copy(_params)
            if cpu_throughput_mode and "CPU" not in params["device"]:
                continue
            if gpu_throughput_mode and "GPU" not in params["device"]:
                continue
            if not gpu_throughput_mode:
                params.pop("gpu_streams", None)
            if not cpu_throughput_mode:
                params.pop("cpu_streams", None)

            name = test.__name__
            test_id = "{}_{}".format(name, "_".join("{}_{}".format(key, val) for (key, val) in sorted(params.items())))

            params.update({"skip_ir_generation": skip_ir_generation})

            try:
                test_case = test(**params, **addit_params_dict, **{"required_params": params}, test_id=test_id)

            except Exception as e:
                tb = traceback.format_exc()
                broken_test = class_factory(cls_name=name, cls_kwargs={"__name__": name, **params, **addit_params_dict,
                                                                       "required_params": params},
                                            BaseClass=BrokenTest)

                test_case = broken_test(test_id=test_id, exception=e,
                                        fail_message="Test {} is broken and fails "
                                                     "with traceback {}".format(name, tb))

            params_for_satisfaction = {"model": name, **params}
            if satisfies_all_rules(params_for_satisfaction, Environment.base_rules, can_partially_match=False) \
                    and not getattr(test_case, "__do_not_run__", False):
                test_ids.append(test_id)
                test_cases.append(test_case)

    if test_cases:
        metafunc.parametrize("instance", test_cases, ids=test_ids)


def pytest_collection_modifyitems(session, config, items):
    """
    Pytest hook for items collection. Adds pytest markers to constructed tests.

    Markers are:
    * Test instance name
    * "Raw" __pytest_marks__ discover in test instances
    * IR generation step parameters (framework, precision)
    * Inference step parameters (inference type, batch, device)
    """

    for i in list(items):
        if not hasattr(i, 'callspec'):
            items.remove(i)

    items.sort(key=lambda item: item.callspec.params['instance'].__class__.__name__)

    logger.info("Preparing tests for test session in the following folder: {}".format(session.startdir))

    deselected = []
    all_components = {}
    all_requirements = {}
    required_marker_ids = hook_utils.get_required_marker_ids_for_test_run()

    pytorch_original_tests = []
    for i in items:
        test_name = i.name.replace(i.originalname, '').replace('[', '').lower()
        pytorch_original_tests.append(test_name.startswith('pytorch_'))

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

            ie_pipeline = getattr(instance, 'ie_pipeline', {})
            ir_gen = ie_pipeline.get('get_ir', {})
            if ir_gen:
                test.add_marker(ir_gen.get('precision', 'FP32'))
            # TODO: Handle marks setting from infer step correctly.
            # TODO: Currently 'network_modifiers' added as mark which is useless
            # infer_step = next(iter(ie_pipeline.get('infer', {}).values()))
            # for name, value in infer_step.items():
            #     mark = '{name}:{value}'.format(name=name, value=value)
            #     # treat bools as flags
            #     if isinstance(value, bool) and value is True:
            #         mark = str(name)
            #     # pass pytest markers and strings "as is"
            #     elif isinstance(value, (type(pytest.mark.Marker), str)):
            #         mark = value
            #     test.add_marker(mark)
        except BrokenTestException as e:
            test.add_marker("broken_test")
            deselected.append(test)
            continue

        test_type = MarkRunType.get_test_type_mark(test)
        hook_utils.update_components(test)
        if hook_utils.deselect(test, test_type, required_marker_ids):
            deselected.append(test)
            continue
        hook_utils.update_markers(test, test_type, all_components, MarkGeneral.COMPONENTS.mark)
        hook_utils.update_markers(test, test_type, all_requirements, MarkGeneral.REQIDS.mark)

    if deselected:
        hook_utils.deselect_items(items, config, deselected)

    # sort items so that we have the sequence of tests being executed as in MarkRunType:
    items[:] = sorted(items, key=lambda element: MarkRunType.test_type_mark_to_int(element))


def call_and_report(item, when, log=True, **kwds):
    import logging as lg
    lg.basicConfig(format="[ %(levelname)s ] %(message)s", level=lg.DEBUG, stream=sys.stdout)
    call = call_runtest_hook(item, when, **kwds)

    hook = item.ihook
    report = hook.pytest_runtest_makereport(item=item, call=call)

    if when == "call" and hasattr(report, "wasxfail"):
        regexp_marks = [m for m in item.own_markers if hasattr(m, "regexps")]
        failed_msgs = {}
        pytest_html = item.config.pluginmanager.getplugin('html')
        extra = getattr(report, 'extra')
        for m in regexp_marks:
            matches = []
            xfail_reason = m.kwargs.get('reason', "UNDEFINED")  # TODO: update for non-xfail marks
            for pattern in m.regexps:
                regexp = re.compile(pattern)
                matches.append(regexp.search(report.caplog) is not None or
                               regexp.search(report.longreprtext) is not None)

            if (m.match_mode == "all" and not all(matches)) or (m.match_mode == "any" and not any(matches)):
                failed_msgs[xfail_reason] = \
                    "Some of regexps '{}' for xfail mark with reason '{}' doesn't match the test log! " \
                    "Test will be forced to fail!".format(', '.join(m.regexps), xfail_reason)
            elif (m.match_mode == "all" and all(matches)) or (m.match_mode == "any" and any(matches)):
                jira_link = "https://jira.devtools.intel.com/browse/{}".format(xfail_reason)
                extra.append(pytest_html.extras.url(jira_link, name=xfail_reason))
                if getattr(item._request, 'test_info', None):
                    item._request.test_info.update({"links2JiraTickets": [xfail_reason]})
                break
        else:
            jira_links = []
            for ticket_num, msg in failed_msgs.items():
                lg.error(msg)
                jira_link = "https://jira.devtools.intel.com/browse/{}".format(ticket_num)
                extra.append(pytest_html.extras.url(jira_link, name=ticket_num))
                jira_links.append(ticket_num)
            report.outcome = "failed"
            if getattr(item._request, 'test_info', None):
                item._request.test_info.update({"links2JiraTickets": jira_links})
            if hasattr(report, "wasxfail"):
                del report.wasxfail
        report.extra = extra

    if log:
        hook.pytest_runtest_logreport(report=report)
    if check_interactive_exception(call, report):
        hook.pytest_exception_interact(node=item, call=call, report=report)

    return report


@pytest.mark.tryfirst
def pytest_runtest_protocol(item, nextitem):
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    # copy of _pytest.runner.runtestprotocol function. Need to use local implementation of call_and_report
    log = True
    hasrequest = hasattr(item, "_request")
    if hasrequest and not item._request:
        item._initrequest()
    rep = call_and_report(item, "setup", log)
    reports = [rep]
    if rep.passed:
        if item.config.option.setupshow:
            show_test_item(item)
        if not item.config.option.setuponly:
            reports.append(call_and_report(item, "call", log))
    reports.append(call_and_report(item, "teardown", log, nextitem=nextitem))
    # after all teardown hooks have been called
    # want funcargs and request info to go away
    if hasrequest:
        item._request = False
        item.funcargs = None
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_sessionfinish(session, exitstatus):
    for dir in Environment.locked_dirs:
        dir_locker = DirLockingHandler(dir)
        dir_locker.unlock()

    if session.config.option.pregen_irs:
        path = (Path(Environment.env['pregen_irs_path']) / session.config.option.pregen_irs).with_suffix('.lock')
        if path.exists():
            os.remove(path)
