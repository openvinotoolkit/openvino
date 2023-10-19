from collections import OrderedDict

from e2e_oss.common_utils.tf_helper import TFVersionHelper
from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.postproc_template import parse_classification
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_tf
from e2e_oss.utils.path_utils import prepend_with_env_path, ref_from_model, resolve_file_path

common_input_file = resolve_file_path("test_data/inputs/tf/classification_imagenet.npz")


class TFLiteClassification(CommonConfig):
    """Base class for TensorFlow Lite classification nets."""
    model = ''
    model_env_key = "tf_internal_models"
    h = 0
    w = 0
    preproc = {}
    postproc = parse_classification()

    def __init__(self, batch, device, precision, api_2, **kwargs):
        preprocess_args = {'batch': batch, 'h': self.h, 'w': self.w}
        if not api_2:
            preprocess_args.update({'permute_order': (2, 0, 1)})

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model, framework="tf")}}),
            ("postprocess", {"align_with_batch": {"batch": batch}, "normalize": {"factor": 255}})])

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc_tf(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version,
                                                             self.model),
                                 precision=precision, input_shape=(1, self.h, self.w, 3)),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])

        self.comparators = classification_comparators(precision=precision, device=device, postproc=self.postproc)
