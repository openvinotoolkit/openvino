import os
from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_pb_input, read_npy_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from utils.path_utils import prepend_with_env_path
from utils.pytest_utils import mark


class ONNXPrecollectedProtobufBaseClass(CommonConfig):
    input_name = "data_0"
    model_env_key = "onnx_internal_models"
    opset = ""
    test_data_set = "test_data_set_0"
    additional_mo_args = {}
    comparator_target_layers = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += tuple([
            mark("onnx", is_simple_mark=True),
            mark("onnx_precollected_data", is_simple_mark=True)
        ])
        self.model = os.path.join(self.opset, self.model)
        model_path = prepend_with_env_path(self.model_env_key, self.model)
        input_path = os.path.join(os.path.dirname(model_path), self.test_data_set, "input_0.pb")
        output_path = os.path.join(os.path.dirname(model_path), self.test_data_set, "output_0.pb")
        outputs_map = {self.output_name: output_path}
        inputs_map = {self.input_name: input_path}

        self.ref_pipeline = OrderedDict([
            read_pb_input(path=outputs_map),
            ("postprocess", {"align_with_batch": {"batch": batch}})
        ])

        self.ie_pipeline = OrderedDict([
            read_pb_input(path=inputs_map),
            ("preprocess", {"align_with_batch": {"batch": batch, "expand_dims": False}}),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision, **self.additional_mo_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])

        self.comparators = eltwise_comparators(precision=precision, device=device,
                                               target_layers=self.comparator_target_layers)


class ONNXPrecollectedNPYBaseClass(CommonConfig):
    input_name = "data_0"
    model_env_key = "onnx_internal_models"
    opset = ""
    test_data_set = "test_data_set_0"
    additional_mo_args = {}
    comparator_target_layers = None
    inputs_map = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += tuple([
            mark("onnx", is_simple_mark=True),
            mark("onnx_precollected_data", is_simple_mark=True)
        ])
        self.model = os.path.join(self.opset, self.model)
        model_path = prepend_with_env_path(self.model_env_key, self.model)
        input_path = os.path.join(os.path.dirname(model_path), self.test_data_set, "input_0.npy")
        output_path = os.path.join(os.path.dirname(model_path), self.test_data_set, "output_0.npy")
        outputs_map = {self.output_name: output_path}
        inputs_map = {self.input_name: input_path}

        self.ref_pipeline = OrderedDict([
            read_npy_input(path=outputs_map),
            ("postprocess", {"align_with_batch": {"batch": batch}})
        ])
        self.ie_pipeline = OrderedDict([
            read_npy_input(path=inputs_map),
            ("preprocess", {"align_with_batch": {"batch": batch, "expand_dims": False}}),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision, **self.additional_mo_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device,
                                               target_layers=self.comparator_target_layers)
