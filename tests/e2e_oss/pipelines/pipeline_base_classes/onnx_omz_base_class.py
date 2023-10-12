from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import reshape_input_shape_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from utils.path_utils import prepend_with_env_path


class ONNX_OMZ_BaseClass(CommonConfig):
    model_env_key = "icv_model_zoo_models"
    use_mo_mapping = False

    def __init__(self, batch, device, precision, api_2, **kwargs):
        model_path = prepend_with_env_path(self.model_env_key, self.model)
        input_file_path = prepend_with_env_path(self.model_env_key, self.input_file)

        self.ref_pipeline = OrderedDict([
            read_npz_input(path=input_file_path),
            ("preprocess", {"align_with_batch": {"batch": 1, "expand_dims": False}}),
            ('get_refs', {'score_onnx_runtime': {'model': model_path,
                                                 "onnx_rt_ep": kwargs.get("onnx_rt_ep", None)}}),
            ("postprocess", {"align_with_batch": {"batch": batch}}),
        ])

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=input_file_path),
            ("preprocess", {"align_with_batch": {"batch": batch, "expand_dims": False}}),
            common_ir_generation(self.environment["mo_runner"],
                                 self.environment["mo_out"],
                                 model_path,
                                 precision,
                                 batch=1),
            reshape_input_shape_infer_step(device=device, input_file_path=input_file_path, api_2=api_2, **kwargs)
        ])
        self.ie_pipeline['get_ir']['mo'].update({"use_input_data_shape": True})
        self.comparators = eltwise_comparators(precision=precision, device=device)
