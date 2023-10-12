from collections import OrderedDict
from pathlib import Path

from common_utils.ir_providers.tf_helper import TFVersionHelper
from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators
from e2e_oss.pipelines.pipeline_templates.comparators_template import segmentation_comparators
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from utils.openvino_resources import OpenVINOResources
from utils.path_utils import prepend_with_env_path
from utils.path_utils import ref_from_model, resolve_file_path
from utils.pytest_utils import mark, timeout

common_input_file = resolve_file_path("test_data/inputs/caffe/cars_road.npz")


class TF_Mask_RCNN_Config(CommonConfig):
    model_env_key = "models"
    model = ''
    mapping = {'tf_detections': 'tf_detections', 'detection_masks': 'detection_masks'}
    json = TFVersionHelper().resolve_tf_transformations_config("mask_rcnn_subgraph_replacement_config_file")
    h = 0
    w = 0

    def __init__(self, batch, device, precision, api_2, **kwargs):
        infer_api = 'ie_sync_api_2' if api_2 else 'ie_sync'
        reshape_api = 'set_batch_using_reshape_api_2' if api_2 else 'set_batch'
        detection_masks = 'SecondStageBoxPredictor_1/Conv_3/BiasAdd'if api_2 else 'masks'

        preprocess_args = {'batch': batch, 'h': self.h, 'w': self.w,
                           'add_layer_to_input_data': {'image_info': [(self.h, self.w, 1)]},
                           'rename_inputs': [('data', 'image_tensor')]}
        if not api_2:
            preprocess_args.update({'permute_order': (2, 0, 1)})

        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True),
                                  mark("od", is_simple_mark=True),
                                  mark("mask", is_simple_mark=True),
                                  mark("tf", is_simple_mark=True),
                                  mark(timeout(1000, "Not enough time for test to finish")))

        model_path = prepend_with_env_path(self.model_env_key,
                                           TFVersionHelper().tf_models_version,
                                           self.model)

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model,
                                                                  framework="tf")}}),
            ("postprocess", OrderedDict([
                ("tf_to_common_od_format", {}),
                ("remove_layer",
                 {"layers_to_remove": ["raw_detection_boxes", "raw_detection_scores",
                                       "detection_multiclass_scores"]}),
                ("align_with_batch_od", {"batch": batch, "target_layers": ["tf_detections"]}),
                ("squeeze", {"axis": 0, "target_layers": ["detection_masks"]}),
                ("align_with_batch", {"batch": batch, "target_layers": ["detection_masks"]}),
                ("parse_object_detection", {"target_layers": ["tf_detections"]})
            ]))])
        pipeline_cfg = str(Path(model_path).with_suffix('.config'))
        mo_cfg = str(OpenVINOResources().mo_extensions / 'front' / 'tf' / self.json)

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision,
                                 input_shape=(1, self.h, self.w, 3),
                                 tensorflow_object_detection_api_pipeline_config=pipeline_cfg,
                                 transformations_config=mo_cfg),
            ('infer', {infer_api: {"device": device, "network_modifiers": {
                reshape_api: {"batch": batch, 'target_layers': ['image_tensor']}},
                       "timeout": 1000}}),
            ("postprocess", OrderedDict([
                ("parse_object_detection", {"target_layers": ['reshape_do_2d']}),
                ("rename_outputs", {"rename_input_pairs": [("reshape_do_2d", "tf_detections"),
                                                           (detection_masks, "detection_masks")]})
            ]))
        ])

        self.comparators = OrderedDict([
            segmentation_comparators.unwrap(target_layers=["score"], precision=precision,
                                            postproc=OrderedDict([
                                                ("parse_mask_rcnn_tf", {"target_layers": [
                                                    "tf_detections", "detection_masks"],
                                                    "h": 800, "w": 800, "num_classes": 90}),
                                                ("parse_semantic_segmentation", {"target_layers": ["score"]})]),
                                            device=device),
            object_detection_comparators.unwrap(target_layers=["tf_detections"], precision=precision,
                                                mean_only_iou=True, p_thr=getattr(self, "p_thr", 0.5),
                                                device=device)])
