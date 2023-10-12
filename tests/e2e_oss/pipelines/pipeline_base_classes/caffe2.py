import os
from collections import OrderedDict

from tests.e2e_oss._utils.path_utils import resolve_file_path, prepend_with_env_path, ref_from_model
from tests.e2e_oss.common_utils.pytest_utils import mark
from tests.e2e_oss.pipelines.pipeline_templates.collect_reference_templates import collect_caffe2_refs
from tests.e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators, eltwise_comparators, \
    segmentation_comparators
from tests.e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from tests.e2e_oss.pipelines.pipeline_templates.postproc_template import parse_classification
from tests.e2e_oss.pipelines.pipeline_templates.postproc_template import parse_semantic_segmentation
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig


class Caffe2ClassificationBaseClass(CommonConfig):
    postproc = parse_classification()
    input_name = "data_0"
    preproc = {}
    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")
    model_env_key = "onnx_internal_models"
    opset = ""

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += (mark("classification", is_simple_mark=True),
                                  mark("caffe2", is_simple_mark=True),
                                  mark("onnx", is_simple_mark=True))

        self.model = os.path.join(self.opset, self.model)
        ref_file = os.path.basename(os.path.splitext(getattr(self, "ref_file_name", self.model))[0] + ".npz")
        model_path = prepend_with_env_path(self.model_env_key, self.model)

        self.ref_collection = collect_caffe2_refs(model=model_path, input=self.input_file, ref_model=ref_file,
                                                  h=self.h, w=self.w, opset=self.opset,
                                                  preprocessors={"rename_inputs": [("data", self.input_name)],
                                                                 "permute_order": (2, 0, 1)}.update(self.preproc))

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(ref_file, "caffe2", self.opset),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(h=self.h, w=self.w, batch=batch, rename_inputs=[("data", self.input_name)],
                             permute_order=(2, 0, 1), **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = classification_comparators(precision=precision, postproc=self.postproc, device=device)


class Caffe2EltwiseBaseClass(CommonConfig):
    input_name = "data_0"
    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")
    model_env_key = "onnx_internal_models"
    postproc = {}
    preproc = {}
    opset = ""

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += tuple([
            mark("caffe2", is_simple_mark=True),
            mark("onnx", is_simple_mark=True)
        ])
        self.model = os.path.join(self.opset, self.model)
        ref_file = os.path.basename(os.path.splitext(getattr(self, "ref_file_name", self.model))[0] + ".npz")
        model_path = prepend_with_env_path(self.model_env_key, self.model)

        self.ref_collection = collect_caffe2_refs(model=model_path, input=self.input_file, ref_model=ref_file,
                                                  h=self.h, w=self.w, opset=self.opset,
                                                  preprocessors={"rename_inputs": [("data", self.input_name)],
                                                                 "permute_order": (2, 0, 1)}.update(self.preproc))

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(ref_file, framework="caffe2", opset=self.opset),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(h=self.h, w=self.w, batch=batch, rename_inputs=[("data", self.input_name)],
                             permute_order=(2, 0, 1), **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device, postproc=self.postproc)


class Caffe2SemanticSegmentationBaseClass(CommonConfig):
    input_name = "data_0"
    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")
    model_env_key = "onnx_internal_models"
    postproc = {}
    preproc = {}
    opset = ""

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += tuple([
            mark("caffe2", is_simple_mark=True),
            mark("onnx", is_simple_mark=True)
        ])
        self.model = os.path.join(self.opset, self.model)
        ref_file = os.path.basename(os.path.splitext(getattr(self, "ref_file_name", self.model))[0] + ".npz")
        model_path = prepend_with_env_path(self.model_env_key, self.model)

        self.ref_collection = collect_caffe2_refs(model=model_path, input=self.input_file, ref_model=ref_file,
                                                  h=self.h, w=self.w, opset=self.opset,
                                                  preprocessors={"rename_inputs": [("data", self.input_name)],
                                                                 "permute_order": (2, 0, 1)}.update(self.preproc))

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(ref_file, "caffe2", self.opset),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(h=self.h, w=self.w, batch=batch, rename_inputs=[("data", self.input_name)],
                             permute_order=(2, 0, 1), **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = segmentation_comparators(postproc=parse_semantic_segmentation(),
                                                    precision=precision, device=device)

        if hasattr(self, 'additional_args'):
            self.ie_pipeline["get_ir"]["mo"]["additional_args"].update({**self.additional_args})
