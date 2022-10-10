# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil

from openvino.tools.mo.utils.ir_engine.ir_engine import IREngine
from openvino.tools.pot.graph.graph_utils import save_graph
from tests.utils.path import REFERENCE_MODELS_PATH


def check_graph(tmp_path, graph, model_name, model_framework, check_weights=False):
    """
    Checking that two graphs are equal by comparing topologies and
    all weights if check_weights is specified as True.
    """
    model_name = '_'.join([model_name, model_framework])
    ir_name_xml = model_name + '.xml'
    path_to_ir_xml = tmp_path.joinpath(ir_name_xml)
    save_graph(graph, tmp_path.as_posix(), model_name, rename_results=False)

    path_to_ref_ir_xml = REFERENCE_MODELS_PATH.joinpath(ir_name_xml)

    if check_weights:
        ir_name_bin = model_name + '.bin'
        path_to_ir_bin = tmp_path.joinpath(ir_name_bin).as_posix()
        path_to_ref_ir_bin = REFERENCE_MODELS_PATH.joinpath(ir_name_bin).as_posix()
    else:
        path_to_ir_bin = None
        path_to_ref_ir_bin = None

    if not path_to_ref_ir_xml.exists():
        shutil.copyfile(path_to_ir_xml.as_posix(), path_to_ref_ir_xml.as_posix())
        if check_weights:
            shutil.copyfile(path_to_ir_bin, path_to_ref_ir_bin)

    ref_graph = IREngine(path_to_ref_ir_xml.as_posix(), path_to_ref_ir_bin)
    test_graph = IREngine(path_to_ir_xml.as_posix(), path_to_ir_bin)

    result, stderr = ref_graph.compare(test_graph)
    if stderr:
        print(stderr)
    assert result


def check_model(tmp_path, model, model_name, model_framework, check_weights=False):
    """
    Checking that graphs of models are equal to their references by comparing topologies and
    all weights if check_weights is specified as True.
    """
    model_name_ = model_name
    for model_dict in model.models:
        if model.is_cascade:
            model_name_ = '{}_{}'.format(model_name, model_dict['name'])
        check_graph(tmp_path, model_dict['model'], model_name_, model_framework, check_weights=check_weights)
