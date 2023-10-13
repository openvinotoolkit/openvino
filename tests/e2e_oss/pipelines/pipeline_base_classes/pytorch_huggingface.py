from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.pytorch_loader_template import pytorch_loader
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from e2e_oss._utils.path_utils import ref_from_model
from e2e_oss.common_utils.pytest_utils import mark


class _PyTorchHuggingFaceBase(CommonConfig):
    """Base class for Pytorch HuggingFace tests"""
    __pytest_marks__ = CommonConfig.__pytest_marks__ + tuple([mark(pytest_mark="pytorch_hf", is_simple_mark=True)])

    # params for pytorch to onnx omz converter script
    model = None
    input_shapes = None
    model_path = None
    import_module = None
    # this line is only need when we want to convert model to onnx,
    # but for now we won't convert big models from hf to onnx format
    output_file = ''
    get_model_args = {}

    # params for pytorch loader
    inputs_order = []
    torch_export_method = None

    input_file = None
    ref_model_path = None
    model_env_key = "pytorch_hf_models_path"
    target_layers = []
    additional_args = {}

    def __init__(self, batch, device, precision, api_2, **kwargs):
        if self.input_shapes:
            self.additional_args.update({'input_shape': [int(x) for x in self.input_shapes.split(',')]})

        self.ref_collection = {"pipeline": OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(convert_to_torch=True),
            ("get_refs", {'score_pytorch_saved_model': {"model": self.model,
                                                        "model-path": self.ref_model_path,
                                                        }}),
            ("postprocess", OrderedDict([
                ("filter_torch_data", {"target_layers": self.target_layers}),
                ("align_with_batch", {"batch": 1})
             ]))
        ]),
            "store_path": ref_from_model(self.model, framework="pytorch"),
            "store_path_for_ref_save": ref_from_model(self.model, framework="pytorch",
                                                      check_empty_ref_path=False)
        }
        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": ref_from_model(model_name=self.model, framework="pytorch")}}),
            ('postprocess', {"align_with_batch": {"batch": batch},
                             "names_to_indices": {}}),
        ])
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(convert_to_torch=True, names_to_indices=True, batch=batch, expand_dims=False),
            pytorch_loader(
                import_module=self.import_module,
                model_name=self.model,
                model_path=self.model_path,
                torch_export_method=self.torch_export_method,
                inputs_order=self.inputs_order
            ),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=self.output_file,
                                 precision=precision,
                                 **self.additional_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
            ("postprocess", {"names_to_indices": {}})
        ])

        self.comparators = eltwise_comparators(device=device, precision=precision)

