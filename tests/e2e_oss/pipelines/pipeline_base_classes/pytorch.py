import os
from collections import OrderedDict

import torch

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.pytorch_loader_template import pytorch_loader
from e2e_oss.pipelines.pipeline_templates.pytorch_to_onnx_converter_template import convert_pytorch_to_onnx
from e2e_oss.pipelines.pipeline_templates.postproc_template import parse_classification
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from e2e_oss._utils.path_utils import prepend_with_env_path, ref_from_model, resolve_file_path
from e2e_oss.common_utils.pytest_utils import mark


class _PyTorchBase(CommonConfig):
    """Base class for Pytorch Pretrained/Torchvision tests"""
    # TODO: (ashchepe) Add dynamic ref_provider choice depending on special cli option
    __pytest_marks__ = CommonConfig.__pytest_marks__ + tuple([mark(pytest_mark="pytorch", is_simple_mark=True)])

    # params for pytorch to onnx omz converter script
    model = None
    pytorch_weights = None
    input_shapes = None
    output_file = ''
    model_path = None
    import_module = None
    input_names = None
    output_names = None
    model_param = {}
    inputs_dtype = 'float'
    conversion_param = None
    opset_version = 11
    converter_timeout = 300

    # params for pytorch loader
    torch_export_method = ''

    h = 224
    w = 224
    get_model_args = {}
    model_env_key = ''
    model_prefix = ''
    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")

    def align_results(self, ref_res, optim_model_res, xml=None):
        cmp_layers = getattr(self, "cmp_layers", None)
        max_result_len = max(len(ref_res), len(optim_model_res))

        if max_result_len != 1 and cmp_layers is None:
            raise KeyError("Multiple output topologies are not supported for PyTorch!")

        assert cmp_layers is None or len(cmp_layers) == 1, \
            "Multiple output topologies are not supported for PyTorch!"

        layer_name = list(optim_model_res.keys())[0] if cmp_layers is None else cmp_layers[0]
        ref_res = {layer_name: list(ref_res.values())[0]}

        return ref_res, optim_model_res

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.batch = batch
        self.output_file = os.path.join(self.environment['mo_out'], f'{self.model}_{self.import_module}.onnx')
        self.torch_model_zoo_path = prepend_with_env_path(self.model_env_key, self.model)
        self.additional_args = {}
        self.onnx_model_path = prepend_with_env_path("pytorch_to_onnx_dump_path", self.model_prefix + self.model + ".onnx")
        if self.convert_pytorch_to_onnx:
            self.input_names = 'input.1'
        if self.pytorch_weights:
            self.pytorch_weights = prepend_with_env_path(self.model_env_key, self.pytorch_weights)
        if self.input_shapes:
            if api_2:
                self.additional_args.update({'example_input': torch.ones([int(x) for x in self.input_shapes.split(',')],
                                                                         dtype=torch.float32)})
            else:
                self.additional_args.update({'input_shape': [int(x) for x in self.input_shapes.split(',')]})

        for attr in ["onnx_rt_ep", "pytorch_models_zoo_path"]:
            val = kwargs.get(attr, None)
            setattr(self, attr, val)

        self.ref_collection = {"pipeline": OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(batch=1,
                             h=self.h,
                             w=self.w,
                             normalization_factor=255,
                             permute_order=(2, 0, 1),
                             rename_inputs=[("data", "input.1")]),
            ("get_refs", {self.ref_provider: {"model_name": self.model,
                                              "torch_model_zoo_path": self.torch_model_zoo_path,
                                              "model": self.model_path,
                                              "onnx_dump_path": self.onnx_model_path,
                                              "onnx_rt_ep": self.onnx_rt_ep,
                                              "convert_to_onnx": False,
                                              "get_model_args": self.get_model_args}}),
            ("postprocess", {"align_with_batch": {"batch": batch}})
        ]),
            "store_path": ref_from_model(self.model, framework="pytorch", opset=self.opset),
            "store_path_for_ref_save": ref_from_model(self.model, framework="pytorch", opset=self.opset,
                                                      check_empty_ref_path=False)
        }
        self.ref_pipeline = read_refs_pipeline(
            ref_file=ref_from_model(model_name=self.model, framework="pytorch", opset=self.opset), batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(h=self.h, w=self.h,
                             batch=batch,
                             normalization_factor=255,
                             permute_order=(2, 0, 1),
                             names_to_indices=True),
            convert_pytorch_to_onnx(
                model_name=self.model,
                weights=self.pytorch_weights,
                input_shapes=self.input_shapes,
                output_file=self.output_file,
                model_path=self.model_path,
                import_module=self.import_module,
                input_names=self.input_names,
                output_names=self.output_names,
                model_param=self.get_model_args,
                inputs_dtype=self.inputs_dtype,
                conversion_param=self.conversion_param,
                opset_version=self.opset_version,
                torch_model_zoo_path=self.torch_model_zoo_path,
                converter_timeout=self.converter_timeout,
            ) if self.convert_pytorch_to_onnx else
            pytorch_loader(
                import_module=self.import_module,
                model_name=self.model,
                model_path=self.model_path,
                model_param=self.get_model_args,
                torch_export_method=self.torch_export_method,
                torch_model_zoo_path=self.torch_model_zoo_path
            ),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=self.output_file,
                                 precision=precision,
                                 **self.additional_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, index_infer=True, **kwargs),
        ])

        cmp_layers = getattr(self, "cmp_layers", None)

        self.comparators = classification_comparators(
            postproc=parse_classification(target_layers=cmp_layers),
            device=device,
            precision=precision,
            target_layers=cmp_layers)


class PyTorchPretrainedBase(_PyTorchBase):
    """Base class for PyTorch Pretrained tests"""

    opset = "pretrained"
    model_env_key = "pytorch_pretrained_models_path"
    model_prefix = "pytorch_pretrained_"
    ref_provider = "score_pytorch_pretrained"


class PyTorchTorchvisionBase(_PyTorchBase):
    """Base class for PyTorch Torchvision tests"""

    opset = "torchvision"
    model_env_key = "pytorch_torchvision_models_path"
    model_prefix = "pytorch_torchvision_"
    ref_provider = "score_pytorch_torchvision"


class PyTorchTorchvisionOpticalFlowBase(_PyTorchBase):
    """Base class for PyTorch Torchvision OpticalFlow tests"""

    opset = "torchvision.optical_flow"
    model_env_key = "pytorch_torchvision_models_path"
    model_prefix = "pytorch_torchvision_optical_flow"
    ref_provider = "score_pytorch_torchvision_optical_flow"


class PyTorchTorchvisionDetectionBase(_PyTorchBase):
    """Base class for PyTorch Torchvision OpticalFlow tests"""

    opset = "torchvision.detection"
    model_env_key = "pytorch_torchvision_models_path"
    model_prefix = "pytorch_torchvision_detection"
    ref_provider = "score_pytorch_torchvision_detection"


class PyTorchTimmBase(_PyTorchBase):
    """Base class for PyTorch Timm tests"""

    opset = "timm"
    model_env_key = "pytorch_timm_models_path"
    model_prefix = "pytorch_timm_"
    ref_provider = "score_pytorch_timm"
