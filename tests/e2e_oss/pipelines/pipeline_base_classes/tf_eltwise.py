from collections import OrderedDict

from tests.e2e_oss.common_utils.tf_helper import TFVersionHelper
from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_tf
from tests.e2e_oss._utils.path_utils import ref_from_model, prepend_with_env_path, resolve_file_path

common_input_file = resolve_file_path("test_data/inputs/tf/classification_imagenet.npz")


class TF_eltwise(CommonConfig):
    """Base class for TensorFlow classification nets."""
    model = ''  # model path
    h = 0  # input image height
    w = 0  # input image width
    preproc = {}  # common input preprocessings (i.e. mean=(1, 2, 3))
    postproc = {}  # postprocessor
    infer_args = {}  # additional IE arguments
    model_env_key = "models"
    input_file = common_input_file
    target_layers = None
    additional_mo_args = {}

    def __init__(self, batch, device, precision, api_2, **kwargs):
        preprocess_args = {'batch': batch, 'h': self.h, 'w': self.w, **self.preproc}
        if not api_2:
            preprocess_args.update({'permute_order': (2, 0, 1)})

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="tf"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_tf(**preprocess_args),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, TFVersionHelper().tf_models_version,
                                                             self.model),
                                 precision=precision, input_shape=(1, self.h, self.w, 3),
                                 **self.additional_mo_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = eltwise_comparators(target_layers=self.target_layers, precision=precision, device=device)
