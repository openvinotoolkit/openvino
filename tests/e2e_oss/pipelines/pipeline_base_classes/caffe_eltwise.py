from collections import OrderedDict
from pathlib import Path

from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from tests.e2e_oss._utils.path_utils import ref_from_model, prepend_with_env_path, resolve_file_path

common_input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc.npz")


class CAFFE_eltwise_Base(CommonConfig):
    model = ''
    h = 0
    w = 0
    preproc = {}
    input_file = common_input_file
    model_env_key = "models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="caffe"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])

        if hasattr(self, 'additional_args'):
            self.ie_pipeline["get_ir"]["mo"]["additional_args"].update({**self.additional_args})

        self.comparators = eltwise_comparators(precision=precision, device=device)
