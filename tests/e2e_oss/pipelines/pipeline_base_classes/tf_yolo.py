import json
from collections import OrderedDict

from common_utils.ir_providers.tf_helper import TFVersionHelper
from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators, eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from utils.openvino_resources import OpenVINOResources
from utils.path_utils import prepend_with_env_path, ref_from_model, resolve_file_path
from utils.pytest_utils import mark

common_input_file = resolve_file_path("test_data/inputs/caffe/car_1_data.npz")


class TF_YoloBase(CommonConfig):
    def __init__(self, *args, **kwargs):
        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True),
                                  mark("od", is_simple_mark=True),
                                  mark("yolo", is_simple_mark=True),
                                  mark("tf", is_simple_mark=True))


class TF_YOLO_V3_Base(TF_YoloBase):
    model = 'yolo_v3_full.pb'
    model_env_key = "tf_internal_models"
    h = 416
    w = 416

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()
        mo_cfg = str(OpenVINOResources().mo_extensions / 'front' / 'tf' / self.mo_yolo_cfg)

        with open(mo_cfg, "r") as cfg:
            yolo_attrs = next(iter(json.load(cfg)))["custom_attributes"]

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)

        preprocess_args = dict(h=self.h, w=self.w, batch=batch, rename_inputs=[('data', self.input_name)],
                               reverse_channels=True)
        if not api_2:
            preprocess_args.update(dict(permute_order=(2, 0, 1)))

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model, framework="tf")}}),
            ("postprocess", OrderedDict([
                ("align_with_batch", {"batch": batch}),
                ("permute_shape", {"order": (0, 3, 1, 2)}),
                ("yolo_region", {"masks_length": len(yolo_attrs["masks"][0]), "classes": yolo_attrs["classes"],
                                 "coords": yolo_attrs["coords"], "do_softmax": False, "num": yolo_attrs["num"]}),
                ("parse_yolo_V3_region", {"classes": yolo_attrs["classes"], "coords": yolo_attrs["coords"],
                                          'input_w': self.w, 'input_h': self.h,
                                          "masks_length": len(yolo_attrs["masks"][0]),
                                          "scale_threshold": 0.001  # scale factor threshold
                                          }),
                ("prob_filter", {"threshold": 0.3}),  # probability threshold for probability filter
                ("nms", {"overlap_threshold": 0.4}),  # IOU threshold in NMS filter
                ("clip_boxes", {"normalized_boxes": True})]))
        ])

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision, input_shape="(1,416,416,3)",
                                 transformations_config=mo_cfg),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
            ("postprocess", OrderedDict([
                ("parse_yolo_V3_region", {"classes": yolo_attrs["classes"], "coords": yolo_attrs["coords"],
                                          'input_w': self.w, 'input_h': self.h,
                                          "masks_length": len(yolo_attrs["masks"][0]),
                                          "scale_threshold": 0.001}),
                ("prob_filter", {"threshold": 0.3}),
                ("nms", {"overlap_threshold": 0.4}),
                ("clip_boxes", {"normalized_boxes": True})]))
        ])
        self.comparators = object_detection_comparators(precision=precision, device=device)


class TF_YOLO_V3_No_Region_Base(TF_YoloBase):
    model_env_key = "tf_internal_models"
    model = ''
    h = 416
    w = 416
    mo_additional_args = {"input_shape": "(1,416,416,3)"}

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)

        preprocess_args = dict(h=self.h, w=self.w, batch=batch, rename_inputs=[('data', self.input_name)],
                               reverse_channels=True)
        postprocess_args = {}
        if not api_2:
            preprocess_args.update(dict(permute_order=(2, 0, 1)))
            postprocess_args = {'permute_shape': {'order': (0, 2, 3, 1)}}

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {
                "path": ref_from_model(model_name=getattr(self, "model_ref_path", self.model), framework="tf")}}),
            ("postprocess", OrderedDict([("align_with_batch", {"batch": batch})]))
        ])

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision, **self.mo_additional_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
            ('postprocess', postprocess_args)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device)


class TF_YOLO_V2_Base(TF_YoloBase):
    model_env_key = "tf_internal_models"
    model = ''
    h = 416
    w = 416
    eps = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()
        mo_cfg = str(OpenVINOResources().mo_extensions / 'front' / 'tf' / self.mo_yolo_cfg)

        with open(mo_cfg, "r") as cfg:
            yolo_attrs = next(iter(json.load(cfg)))["custom_attributes"]

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)

        preprocess_args = dict(h=self.h, w=self.w, batch=batch, rename_inputs=[('data', 'input')],
                               reverse_channels=True, normalization_factor=255)
        if not api_2:
            preprocess_args.update(dict(permute_order=(2, 0, 1)))

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model, framework="tf")}}),
            ("postprocess", OrderedDict([
                ("align_with_batch", {"batch": batch}),
                ("permute_shape", {"order": (0, 3, 1, 2)}),
                ("yolo_region", {"classes": yolo_attrs["classes"], "coords": yolo_attrs["coords"],
                                 "do_softmax": yolo_attrs["do_softmax"], "num": yolo_attrs["num"]}),
                ("parse_yolo_V2_region", {"classes": yolo_attrs["classes"], "coords": yolo_attrs["coords"],
                                          "num": yolo_attrs["num"], "anchors": yolo_attrs["anchors"],
                                          "grid": (13, 13)}),
                ("prob_filter", {"threshold": 0.3}),
                ("nms", {"overlap_threshold": 0.4}),
                ("clip_boxes", {"normalized_boxes": True})]))
        ])

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision, input_shape="(1,416,416,3)",
                                 transformations_config=mo_cfg),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
            ("postprocess", OrderedDict([
                ("parse_yolo_V2_region", {"classes": yolo_attrs["classes"], "coords": yolo_attrs["coords"],
                                          "num": yolo_attrs["num"], "anchors": yolo_attrs["anchors"],
                                          "grid": (13, 13)}),
                ("prob_filter", {"threshold": 0.3}),
                ("nms", {"overlap_threshold": 0.4}),
                ("clip_boxes", {"normalized_boxes": True})]))
        ])
        self.comparators = object_detection_comparators(precision=precision, p_thr=getattr(self, "p_thr", 0.5),
                                                        a_eps=self.eps, r_eps=self.eps, device=device)


class TF_YOLO_V2_Full_No_Region_Base(TF_YoloBase):
    model_env_key = "tf_internal_models"
    model = ''
    h = 416
    w = 416
    score_tf_args = {}
    mo_additional_args = {"input_shape": "(1,416,416,3)"}

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)
        preprocess_args = dict(h=self.h, w=self.w, batch=batch, rename_inputs=[('data', 'input')],
                               reverse_channels=True, normalization_factor=255)
        postprocess_args = {}
        if not api_2:
            preprocess_args.update(dict(permute_order=(2, 0, 1)))
            postprocess_args = OrderedDict([("permute_shape", {"order": (0, 2, 3, 1)})])

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(
                model_name=getattr(self, "model_ref_path", self.model), framework="tf")}}),
            ("postprocess", OrderedDict([("align_with_batch", {"batch": batch})]))
        ])
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision, **self.mo_additional_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
            ('postprocess', postprocess_args)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device)
