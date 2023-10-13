import os
from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import get_refs_onnx_runtime
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators, eltwise_comparators, \
    dummy_comparators, object_detection_comparators, ssim_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.postproc_template import parse_classification, parse_object_detection, \
    parse_image_modification
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from e2e_oss._utils.path_utils import prepend_with_env_path, resolve_file_path
from e2e_oss.common_utils.pytest_utils import mark


class ONNXRuntimeEltwiseBaseClass(CommonConfig):
    input_name = "data_0"
    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")
    model_env_key = "onnx_internal_models"
    h = 0  # input height
    w = 0  # input width
    postproc = {}
    preproc = {}
    opset = ""

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += tuple([
            mark("onnx_runtime", is_simple_mark=True),
            mark("onnx", is_simple_mark=True),
        ])
        self.onnx_rt_ep = kwargs.get('onnx_rt_ep', None)
        self.model = os.path.join(self.opset, self.model)
        model_path = prepend_with_env_path(self.model_env_key, self.model)

        self.ref_pipeline = OrderedDict([
            read_npz_input(self.input_file),
            assemble_preproc(batch=1, h=self.h, w=self.w,
                             permute_order=(2, 0, 1),
                             rename_inputs=[("data", self.input_name)],
                             **self.preproc),
            get_refs_onnx_runtime(model=model_path, onnx_rt_ep=self.onnx_rt_ep),
            ("postprocess", OrderedDict([("align_with_batch", {"batch": batch})])),
        ])
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(h=self.h, w=self.w, batch=batch, rename_inputs=[("data", self.input_name)],
                             permute_order=(2, 0, 1), **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path, precision=precision,
                                 **getattr(self, "additional_mo_args", {})),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device, postproc=self.postproc,
                                               a_eps=getattr(self, "a_eps", None), r_eps=getattr(self, "r_eps", None))


class ONNXRuntimeClassificationBaseClass(ONNXRuntimeEltwiseBaseClass):
    postproc = parse_classification()

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch, device, precision, api_2, **kwargs)
        self.__pytest_marks__ += ONNXRuntimeEltwiseBaseClass.__pytest_marks__ + tuple([
            mark("classification", is_simple_mark=True)
        ])
        self.comparators = classification_comparators(precision=precision, postproc=self.postproc, device=device,
                                                      a_eps=getattr(self, "a_eps", None),
                                                      r_eps=getattr(self, "r_eps", None))


class ONNXRuntimeObjectDetectionBaseClass(ONNXRuntimeEltwiseBaseClass):
    __pytest_marks__ = ONNXRuntimeEltwiseBaseClass.__pytest_marks__ + tuple([
        mark("object_detection", is_simple_mark=True)
    ])
    postproc = parse_object_detection()

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch, device, precision, api_2, **kwargs)
        self.comparators = object_detection_comparators(precision=precision, postproc=self.postproc, device=device,
                                                        a_eps=getattr(self, "a_eps", None),
                                                        r_eps=getattr(self, "r_eps", None))


class ONNXRuntimeStyleTransferBaseClass(ONNXRuntimeEltwiseBaseClass):
    __pytest_marks__ = ONNXRuntimeEltwiseBaseClass.__pytest_marks__ + tuple([
        mark("style_transfer", is_simple_mark=True)
    ])
    postproc = parse_image_modification()

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch, device, precision, api_2, **kwargs)
        self.comparators = ssim_comparators(precision=precision, postproc=self.postproc, device=device)


class ONNXRuntimeNoComparisonBaseClass(ONNXRuntimeEltwiseBaseClass):
    __pytest_marks__ = ONNXRuntimeEltwiseBaseClass.__pytest_marks__ + tuple([
        mark("dummy", is_simple_mark=True)
    ])
    align_results = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch, device, precision, api_2, **kwargs)
        self.ie_pipeline = OrderedDict()
        self.comparators = dummy_comparators()
