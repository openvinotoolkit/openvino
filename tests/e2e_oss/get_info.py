"""
get_info.py is supposed to be used to obtain information from e2e tests.
This file was created because reference collection and IE pipeline execution are separated,
but it combined here to write all info in one file through one test run.

Default run:
$ pytest get_info.py

Options[*]:
--modules       Paths to tests
--env_conf      Path to environment config
--test_conf     Path to test config
--base_rules_conf    Path to test rules config
--reshape_rules_conf    Path to reshape test rules config 
--dynamism_rules_conf    Path to dynamism test rules config 


[*] For more information see conftest.py
"""
# pylint:disable=invalid-name
import logging as log
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from flaky import flaky

from e2e_oss.extractors import loader_pipeline
from e2e_oss._utils.get_test_info import TestInfo
from utils.e2e.env_tools import Environment
from utils.e2e.ir_provider.model_optimizer_runner import MORunner
from utils.multiprocessing_utils import multiprocessing_run
from utils.pytest_utils import timeout_error_filter
from openvino.tools.mo.convert_impl import arguments_post_parsing
from openvino.frontend import FrontEndManager

if platform.system() == "Darwin":
    import multiprocessing

    multiprocessing.set_start_method("forkserver", True)

pytest_plugins = ('e2e_oss.plugins.e2e_test.conftest',)


def get_data(argv):
    info = TestInfo()
    error_msg = ''

    try:
        sys.path.extend(str(Environment.env['mo_root']))
        from openvino.tools.mo.utils.cli_parser import get_all_cli_parser
        from openvino.tools.mo.utils.logger import init_logger
        init_logger('ERROR', True)

        parser = get_all_cli_parser()

        for i in range(len(argv)):
            if i % 2 == 0:
                argv[i] = f'--{argv[i]}'

        argv = parser.parse_args(args=argv)
        argv.feManager = FrontEndManager()
        argv = arguments_post_parsing(argv)
        graph = loader_pipeline(argv)
        nodes = graph.get_op_nodes()

        for node in nodes:
            name = node.soft_get('name', node.id)
            op = node.op
            shape = node.soft_get('shape', node.soft_get('shapes'))
            shape = [int(x) for x in re.findall(r'-?\d+', str(shape))]
            dtype = node.soft_get('data_type')
            if isinstance(dtype, np.dtype):
                dtype = dtype.name

            elif dtype is np.float32:
                dtype = dtype.__name__
            elif dtype is np.int32:
                dtype = 'int32'

            if len(node.in_nodes()) == 0:
                if node.has_valid('value') or node.soft_get('op') == 'Const':
                    op_type = 'Const'
                    info.fill_extra_info(op_type, name, op, shape, dtype)
                elif node.soft_get('op') == 'FakeConst':
                    if 'Variable' in name:
                        op_type = 'Variable'
                        info.fill_extra_info(op_type, name, op, shape, dtype)
                    else:
                        op_type = 'FakeConst'
                        info.fill_extra_info(op_type, name, op, shape, dtype)
                else:
                    op_type = 'Input'
                    info.fill_extra_info(op_type, name, op, shape, dtype)

            else:
                if len(node.out_nodes()) == 0:
                    op_type = 'Output'
                    info.fill_extra_info(op_type, name, op, shape, dtype)
                else:
                    op_type = 'Intermediate'
                    info.fill_extra_info(op_type, name, op, shape, dtype)
        info.extra_info['framework'] = {'name': argv.framework}
    except Exception:
        raise
    return error_msg, info


@flaky(max_runs=3, min_passes=1, rerun_filter=timeout_error_filter)
def test_run(instance, dir_to_dump):
    """Parameterized test.

    :param instance: test instance
    :param dir_to_dump: directory where .yaml file should be dumped
    """
    assert dir_to_dump, "Please, specify path to folder where get_info.py should dump model.yml file"

    get_mo_params = instance.ie_pipeline.get('get_ir', {})
    assert 'mo' in get_mo_params, 'No info for MO'

    instance_name = str(instance).split('.')[1].split(' ')[0]

    extensions = str(Path('_utils/extensions'))
    get_mo_params.get('additional_args', {})
    get_mo_params['mo']['additional_args'].update({'extensions': extensions})

    os.environ['MO_ENABLED_TRANSFORMS'] = ','.join(
        ['TFPrivateExtractor', 'ONNXPrivateExtractor', 'MxNetPrivateExtractor', 'CaffePrivateExtractor',
         'KaldiPrivateExtractor'])

    argv = MORunner(get_mo_params['mo'])._prepare_command_line()[2:]

    error_message, info = multiprocessing_run(func=get_data,
                                              func_args=[argv],
                                              func_log_name='Getting info about original model',
                                              timeout=500)

    if error_message:
        raise RuntimeError("\nMO running failed: \n{}".format(error_message))
    info = TestInfo()
    if 'mo' in get_mo_params:
        # info.extra_info['name'] = str(instance).split('.')[0][1:]
        # TODO currently there are no tests in e2e with --extensions key, but it could occur in future
        redundant_params = ['mo_out', 'mo_runner', 'extensions']
        for param, value in get_mo_params['mo'].items():
            if param not in redundant_params:
                info.fill_mo_args(**{param: value})

    model_path = info.extra_info['model_optimizer_args']['model']
    model_rel_path = os.path.relpath(model_path, os.environ.get('SHARE', Path(model_path).root))
    file_to_dump = Path(dir_to_dump) / Path(model_rel_path).parent / 'model.yml'

    if info.extra_info['model_optimizer_args'].get('extensions'):
        del info.extra_info['model_optimizer_args']['extensions']

    info.extra_info['model_optimizer_args']['model'] = Path(model_path).name
    info.extra_info['model_optimizer_args'] = ['--{}={}'.format(k, v) if v not in ['True', 'False']
                                               else '--{}'.format(k) for k, v in
                                               info.extra_info['model_optimizer_args'].items()]

    info.e2e_models_info = dict(info.e2e_models_info)
    info.extra_info = dict(info.extra_info)
    instance_info = info.extra_info
    info.e2e_models_info.update({instance_name: instance_info})

    file_to_dump.parent.mkdir(parents=True, exist_ok=True)

    with open(file_to_dump, 'w') as file:
        yaml.dump(info.e2e_models_info, file)
        log.info('Dumping test info into {}'.format(file_to_dump))
