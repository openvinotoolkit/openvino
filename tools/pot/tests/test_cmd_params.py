# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino.tools.pot.app.argparser import get_common_argument_parser, check_dependencies
from openvino.tools.pot.app.run import  _update_config_path
from openvino.tools.pot.configs.config import Config
from tests.utils.path import TOOL_CONFIG_PATH, ENGINE_CONFIG_PATH


def check_wrong_parametrs(argv):
    parser = get_common_argument_parser()
    args = parser.parse_args(args=argv)
    check_dependencies(args)
    if not args.config:
        _update_config_path(args)


test_params = [('', 'Either --config or --quantize option should be specified', ValueError),
               ('-e -m path_model', 'Either --config or --quantize option should be specified', ValueError),
               ('--quantize default -w path_weights -m path_model',
                '--quantize option requires AC config to be specified '
                'or --engine should be `simplified`.', ValueError),
               ('--quantize accuracy_aware -m path_model --ac-config path_config',
                '--quantize option requires model and weights to be specified.', ValueError),
               ('-c path_config -m path_model', 'Either --config or --model option should be specified', ValueError),
               ('--quantize default -w path_weights -m path_model --engine simplified',
                'For Simplified mode `--data-source` option should be specified', ValueError),
               ]
@pytest.mark.parametrize('st, match, error', test_params,
                         ids=['{}_{}_{}'.format(v[0], v[1], v[2]) for v in test_params])
def test_wrong_parametrs_cmd(st, match, error):
    with pytest.raises(error, match=match):
        check_wrong_parametrs(st.split())


TOOL_CONFIG_NAME = [('mobilenet-v2-pytorch_single_dataset.json', '-q default -w path_w -m path_m --ac-config path_ac')]


@pytest.mark.parametrize(
    'config_name, argv', TOOL_CONFIG_NAME,
    ids=['{}_{}'.format(v[0], v[1]) for v in TOOL_CONFIG_NAME]
)
def test_load_tool_config(config_name, argv):

    parser = get_common_argument_parser()
    argv = argv.split()
    argv[-1] = ENGINE_CONFIG_PATH.joinpath('mobilenet-ssd.json').as_posix()
    args = parser.parse_args(args=argv)
    tool_config_path = TOOL_CONFIG_PATH.joinpath(config_name).as_posix()
    config = Config.read_config(tool_config_path)
    config.configure_params()
    config.update_from_args(args)
    assert config.model.model == argv[5]
    assert config.model.weights == argv[3]
