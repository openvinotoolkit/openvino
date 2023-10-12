import os
from collections import OrderedDict

import numpy as np

from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import batch_reshape_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_mxnet
from tests.e2e_oss._utils.path_utils import prepend_with_env_path, resolve_file_path, ref_from_model
from tests.e2e_oss.common_utils.pytest_utils import mark


class MXNET_SSD_Base(CommonConfig):
    h = 512
    w = 512
    r_eps = None
    a_eps = None
    input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc.npz")
    preproc = {"mean": (123, 117, 104)}
    postproc = OrderedDict([("parse_object_detection", {}),
                            ("classes_filter", {"classes": [-1]})])
    model_env_key = "mxnet_internal_models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True),
                                  mark("od", is_simple_mark=True),
                                  mark("mxnet", is_simple_mark=True))

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=os.path.splitext(self.model)[0][:-5],
                                                                  framework="mxnet")}}),
            ("postprocess", OrderedDict([("mxnet_to_common_od_format", {"target_layers": ["detection"]}),
                                         ("align_with_batch_od", {"batch": batch})]))])
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_mxnet(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision,
                                 input_shape=(1, 3, self.h, self.w),
                                 legacy_mxnet_model=True),
            batch_reshape_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        if getattr(self, "iou_thr", None):
            self.comparators = object_detection_comparators(postproc=self.postproc, precision=precision,
                                                            r_eps=self.r_eps, a_eps=self.a_eps,
                                                            iou_thr=self.iou_thr, device=device)
        else:
            self.comparators = object_detection_comparators(postproc=self.postproc, precision=precision,
                                                            r_eps=self.r_eps, a_eps=self.a_eps, device=device)


class MXNET_SSD_GluonCV_Base(CommonConfig):
    h = 512
    w = 512
    r_eps = None
    a_eps = None
    p_thr = None
    mean_iou_only = False
    input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc.npz")
    postproc = OrderedDict([("parse_object_detection", {})])
    model_env_key = "mxnet_internal_models"

    @staticmethod
    def concat_outputs_for_SSD_GluonCV():
        def parse_outputs(data: dict):
            data_keys = sorted(data.keys(), reverse=False)
            output_layers = [data[key] for key in data_keys]
            concatenate_data = np.concatenate(output_layers, axis=2)
            return dict(DetectionOutput=concatenate_data)
        return parse_outputs

    def align_results(self, ref_res, optim_model_res, xml=None):
        if len(ref_res) > 1 or len(optim_model_res) > 1:
            raise KeyError("Multiple output topologies are not supported!")
        ref_key_name = list(ref_res.keys())[0]
        ref_res[list(optim_model_res.keys())[0]] = ref_res[ref_key_name]
        ref_res.pop(ref_key_name, None)
        return ref_res, optim_model_res

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(
                model_name=os.path.splitext(self.model)[0][:-5] + '.params', framework="mxnet")}}),
            ("postprocess", OrderedDict([
                ("custom_postproc", {"execution_function": self.concat_outputs_for_SSD_GluonCV()}), (
                    "mxnet_to_common_od_format", {"target_layers": ["DetectionOutput"]}),
                ("align_with_batch_od", {"batch": batch})]))
        ])
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_mxnet(batch=batch, h=self.h, w=self.w),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision,
                                 input_shape=(1, 3, self.h, self.w),
                                 legacy_mxnet_model=True,
                                 enable_ssd_gluoncv=True),
            batch_reshape_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        if getattr(self, "iou_thr", None):
            self.comparators = object_detection_comparators(postproc=self.postproc, precision=precision,
                                                            r_eps=self.r_eps, a_eps=self.a_eps,
                                                            iou_thr=self.iou_thr,
                                                            p_thr=self.p_thr, mean_only_iou=self.mean_iou_only,
                                                            device=device)
        else:
            self.comparators = object_detection_comparators(postproc=self.postproc, precision=precision,
                                                            r_eps=self.r_eps, a_eps=self.a_eps,
                                                            p_thr=self.p_thr, mean_only_iou=self.mean_iou_only,
                                                            device=device)
