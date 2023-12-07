"""Common reference collection templates processed by testing framework.
"""
import os
from collections import OrderedDict
from pathlib import Path

from e2e_oss.common_utils.decorators import wrap_ord_dict
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import ir_pregenerated
from e2e_oss.pipelines.pipeline_templates.postproc_template import assemble_postproc_tf
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc, assemble_preproc_tf
from e2e_oss.utils.path_utils import proto_from_model, ref_from_model, symbol_from_model


def get_refs_onnx_runtime(model, onnx_rt_ep, cast_input_data=True, cast_type="float32"):
    """
    Construct ONNX Runtime reference collection action.

    :param model: .onnx file
    :param onnx_rt_ep: execution provider to infer model
    :param cast_input_data: whether cast or not input data to specific dtype
    :param cast_type: type of data model input data cast to
    :return: ONNX models "get_refs" action processed by testing framework
    """
    return 'get_refs', {'score_onnx_runtime': {'model': model, 'onnx_rt_ep': onnx_rt_ep,
                                               'cast_input_data': cast_input_data,
                                               'cast_input_data_to_type': cast_type}}


def get_refs_paddlepaddle(model, params_filename=None):
    """
    Construct PaddlePaddle reference collection action.

    :param model: model file path which will be used in get_model() function
    :param params_filename: the name of single binary file to load all model parameters.
    :return: PaddlePaddle models "get_refs" action processed by testing framework
    """
    return 'get_refs', {'score_paddlepaddle': {'model': model, 'params_filename': params_filename}}


def get_refs_tf(model=None, output_nodes_for_freeze=None, additional_outputs=[], additional_inputs=[],
                override_default_outputs=False, override_default_inputs=False, saved_model_dir=None,
                user_output_node_names_list=[], score_class_name="score_tf"):
    """
    Construct TensorFlow reference collection action.

    :param model: .pb or .meta file with model
    :param output_nodes_for_freeze: output nodes used for freeze input model before inference
    :param score_class_name: "score_tf", "score_tf_dir", "score_tf_meta" - type of loading model
    :return: TF models "get_refs" action processed by testing framework
    """
    return "get_refs", {score_class_name: {"model": model,
                                           "saved_model_dir": saved_model_dir,
                                           "output_nodes_for_freeze": output_nodes_for_freeze,
                                           "additional_outputs": additional_outputs,
                                           "additional_inputs": additional_inputs,
                                           "override_default_outputs": override_default_outputs,
                                           "override_default_inputs": override_default_inputs,
                                           "user_output_node_names_list": user_output_node_names_list}}


@wrap_ord_dict
def read_refs_pipeline(ref_file, batch):
    """
    Construct Read pre-collected references pipeline

    :param ref_file: path to .npz file with pre-collected references
    :param batch: target batch size
    :return: OrderedDict with pipeline containing get_refs and postprocess steps
    """
    return [("get_refs", {"precollected": {"path": ref_file}}),
            ("postprocess", {"align_with_batch": {"batch": batch}})]


@wrap_ord_dict
def read_pytorch_refs_pipeline(ref_file, batch):
    """
    Construct Read pre-collected references pipeline

    :param ref_file: path to .npz file with pre-collected references
    :param batch: target batch size
    :return: OrderedDict with pipeline containing get_refs and postprocess steps
    """
    return [("get_refs", {"torch_precollected": {"path": ref_file}}),
            ("postprocess", {"align_with_batch": {"batch": batch}})]


@wrap_ord_dict
def read_tf_refs_pipeline(ref_file, batch=None, align_with_batch_od=False, postprocessors=None):
    """
    Construct Read pre-collected references pipeline

    :param ref_file: path to pre-collected references
    :param batch: target batch size
    :param align_with_batch_od: batch alignment preprocessor
    :return: OrderedDict with pipeline containing get_refs and postprocess steps
    """
    if postprocessors is None:
        postprocessors = {}
    return [("get_refs", {"precollected": {"path": ref_from_model(ref_file, framework="tf")}}),
            assemble_postproc_tf(batch=batch, align_with_batch_od=align_with_batch_od, **postprocessors)]


def collect_paddlepaddle_refs(model, input, h=None, w=None, params_filename=None, ref_name=None, preprocessors=None):
    """Construct PaddlePaddle reference collection pipeline."""
    if preprocessors is None:
        preprocessors = {}
    return {'pipeline': OrderedDict([
        read_npz_input(input),
        assemble_preproc(batch=1, **preprocessors) if (h and w) is None
        else assemble_preproc(batch=1, h=h, w=w, permute_order=(2, 0, 1), **preprocessors),
        get_refs_paddlepaddle(model=model,
                              params_filename=params_filename)
    ]),
        'store_path': ref_from_model(ref_name, framework="paddlepaddle"),
        'store_path_for_ref_save': ref_from_model(ref_name, framework="paddlepaddle", check_empty_ref_path=False)}


def collect_tf_refs_pipeline(model, input, saved_model_dir=None, h=None, w=None, ref_name=None, preprocessors=None):
    """Construct reference collection pipeline."""
    if preprocessors is None:
        preprocessors = {}
    if not ref_name:
        ref_name = model
    return {'pipeline': OrderedDict([
        read_npz_input(input),
        assemble_preproc_tf(batch=1, **preprocessors) if (h and w) is None
        else assemble_preproc_tf(batch=1, h=h, w=w, **preprocessors),
        get_refs_tf(model=model) if saved_model_dir is None
        else get_refs_tf(saved_model_dir=saved_model_dir)
    ]),
        'store_path': ref_from_model(ref_name, framework="tf"),
        'store_path_for_ref_save': ref_from_model(ref_name, framework="tf", check_empty_ref_path=False)}


def collect_onnx_refs_pipeline(model, input, onnx_rt_ep, framework, cast_type="float32", h=None, w=None,
                               ref_name=None, preprocessors=None, batch=1, cast_input_data=True):
    """Construct reference collection pipeline."""
    if preprocessors is None:
        preprocessors = {}
    if not ref_name:
        ref_name = model
    return {'pipeline': OrderedDict([
        read_npz_input(input),
        assemble_preproc(batch=batch, **preprocessors) if (h and w) is None
        else assemble_preproc(batch=batch, h=h, w=w, **preprocessors),
        get_refs_onnx_runtime(model=model, onnx_rt_ep=onnx_rt_ep, cast_type=cast_type, cast_input_data=cast_input_data)
    ]),
        'store_path': ref_from_model(ref_name, framework=framework),
        'store_path_for_ref_save': ref_from_model(ref_name, framework=framework, check_empty_ref_path=False)}


def get_refs_tf_hub():
    """
    Construct TensorFlow Hub reference collection action.
    """
    return "get_refs_tf_hub", {'score_tf_hub': {}}
