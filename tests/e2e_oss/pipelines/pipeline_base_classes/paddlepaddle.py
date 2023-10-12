from collections import OrderedDict

from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline, \
    collect_paddlepaddle_refs
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators, \
    object_detection_comparators, segmentation_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.postproc_template import paddlepaddle_od_postproc
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from tests.e2e_oss._utils.path_utils import prepend_with_env_path, ref_from_model, resolve_file_path


class PaddlePaddleEltwise(CommonConfig):
    """Base class for PaddlePaddle tests"""
    use_mo_mapping = False
    params_filename = None
    comparator_target_layers = None

    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")
    model_env_key = "paddlepaddle_internal_models"
    input_name = "inputs"
    h = 227  # input height
    w = 227  # input width
    preproc = {}
    mo_additional_args = {}

    def __init__(self, batch, device, precision, api_2, skip_ir_generation, **kwargs):
        inputs_shapes = [f"{input_name}[{' '.join(map(str, shapes['default_shape']))}]"
                         for input_name, shapes in self.input_descriptor.items()]

        self.ref_collection = collect_paddlepaddle_refs(
            model=prepend_with_env_path(self.model_env_key, self.model),
            params_filename=self.params_filename,
            input=self.input_file,
            h=self.h, w=self.w,
            ref_name=self.ref_name,
            preprocessors={"rename_inputs": [("data", self.input_name)], **self.preproc})

        self.ref_pipeline = read_refs_pipeline(
            ref_file=ref_from_model(model_name=self.ref_name, framework="paddlepaddle"), batch=batch)

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(batch=batch, h=self.h, w=self.w,
                             permute_order=(2, 0, 1),
                             rename_inputs=[("data", self.input_name)],
                             **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 input=self.mo_additional_args.pop("input", ','.join(inputs_shapes)),
                                 precision=precision,
                                 **self.mo_additional_args),
            common_infer_step(device=device, batch=batch, api_2=api_2, skip_ir_generation=skip_ir_generation,
                              input_file_path=self.input_file, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device,
                                               target_layers=self.comparator_target_layers)


class PaddlePaddleOD(PaddlePaddleEltwise):
    """Base class for PaddlePaddle Object Detection tests"""
    input_file = resolve_file_path("test_data/inputs/onnx/VOC_2012_001863_normalised.npz", as_str=True)

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch=batch, device=device, precision=precision, api_2=api_2, **kwargs)
        self.comparators = object_detection_comparators(
            precision=precision, device=device,
            postproc=paddlepaddle_od_postproc(self.comparator_target_layers))


class PaddlePaddleSS(PaddlePaddleEltwise):
    """Base class for PaddlePaddle Semantic Segmentation tests"""
    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch=batch, device=device, precision=precision, api_2=api_2, **kwargs)
        self.comparators = segmentation_comparators(precision=precision, device=device,
                                                    target_layers=self.comparator_target_layers)
