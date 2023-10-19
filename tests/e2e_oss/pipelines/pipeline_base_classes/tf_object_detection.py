from collections import OrderedDict
from pathlib import Path

from e2e_oss.common_utils.openvino_resources import OpenVINOResources
from e2e_oss.common_utils.tf_helper import TFVersionHelper
from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.postproc_template import parse_object_detection
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_tf
from e2e_oss.utils.path_utils import prepend_with_env_path, resolve_file_path, ref_from_model
from e2e_oss.common_utils.pytest_utils import mark

common_input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc_people.npz")


class TF_OD_Config(CommonConfig):
    model = ''  # model path
    h = 0  # input image height
    w = 0  # input image width
    input_file = common_input_file
    preproc = {}  # common input preprocessings (i.e. mean=(1, 2, 3))
    postproc = {}  # postprocessor
    json = ''  # mo tf custom operations config
    model_env_key = "models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        infer_api = 'ie_sync_api_2' if api_2 else 'ie_sync'
        reshape_api = 'set_batch_using_reshape_api_2' if api_2 else 'set_batch'
        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True),
                                  mark("od", is_simple_mark=True),
                                  mark("tf", is_simple_mark=True))

        model_path = prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version, self.model)
        preprocess_args = {'batch': batch, 'h': self.h, 'w': self.w, 'rename_inputs': [('data', 'image_tensor')]}
        if not api_2:
            preprocess_args.update({'permute_order': (2, 0, 1)})

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model, framework="tf")}}),
            ("postprocess", OrderedDict([
                ("tf_to_common_od_format", {}),
                ("remove_layer", {"layers_to_remove": ["raw_detection_boxes", "raw_detection_scores",
                                                       "detection_multiclass_scores"]}),
                ("align_with_batch_od", {"batch": batch})]))])

        pipeline_cfg = str(Path(model_path).with_suffix('.config'))
        mo_cfg = str(OpenVINOResources().mo_extensions / 'front' / 'tf' / self.json)

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_tf(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision,
                                 input_shape=(1, self.h, self.w, 3),
                                 tensorflow_object_detection_api_pipeline_config=pipeline_cfg,
                                 transformations_config=mo_cfg),
            ('infer', {infer_api: {"device": device,
                                   "network_modifiers": {reshape_api: {"batch": batch,
                                                                       "target_layers": ["image_tensor"]}}}}),
        ])
        self.comparators = object_detection_comparators(postproc=parse_object_detection(), precision=precision,
                                                        device=device, p_thr=getattr(self, "p_thr", 0.5))
