from collections import OrderedDict

from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.postproc_template import parse_object_detection
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from tests.e2e_oss._utils.path_utils import prepend_with_env_path, ref_from_model, resolve_file_path
from tests.e2e_oss.common_utils.pytest_utils import mark

common_input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc.npz")


class CAFFE_OD_Base(CommonConfig):
    model = ''
    model_env_key = "models"
    h = 0
    w = 0
    preproc = {}
    input_file = common_input_file
    additional_mo_args = {}

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True, target_runner="test_run"),
                                  mark("od", is_simple_mark=True, target_runner="test_run"),
                                  mark("caffe", is_simple_mark=True, target_runner="test_run"))
        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model, framework="caffe")}}),
            ("postprocess", {"align_with_batch_od": {"batch": batch}})])
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision, **self.additional_mo_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = object_detection_comparators(precision=precision, device=device,
                                                        postproc=parse_object_detection())
