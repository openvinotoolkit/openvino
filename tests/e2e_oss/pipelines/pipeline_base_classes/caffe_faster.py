from collections import OrderedDict

from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import dummy_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from tests.e2e_oss._utils.path_utils import prepend_with_env_path, resolve_file_path

common_input_file = resolve_file_path("test_data/inputs/caffe/faster_rcnn_base.npz")


class CAFFE_Faster_RCNN_Base(CommonConfig):
    model_env_key = "models"
    model = ''
    h = 0
    w = 0
    align_results = None
    target_layers = ['data']

    def __init__(self, batch, device, precision, api_2, **kwargs):
        infer_api = 'ie_sync_api_2' if api_2 else 'ie_sync'
        network_modifier_api = 'set_batch_using_reshape_api_2' if api_2 else 'set_batch_using_reshape'

        self.ref_pipeline = OrderedDict()
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, target_layers=self.target_layers,
                                   reverse_channels=True),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision),
            ("infer", {infer_api: {"device": device,
                                   "network_modifiers": {network_modifier_api: {"batch": batch,
                                                                                "target_layers": self.target_layers}}}})
        ])
        self.comparators = dummy_comparators()
