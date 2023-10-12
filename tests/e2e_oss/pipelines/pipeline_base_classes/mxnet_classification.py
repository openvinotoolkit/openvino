from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import collect_mxnet_refs_pipeline, \
    read_mxnet_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.postproc_template import parse_classification
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_mxnet
from utils.path_utils import prepend_with_env_path, resolve_file_path
from utils.pytest_utils import mark


class MXNET_ClassificationNet(CommonConfig):
    """Base class for MXNet classification nets."""
    ref_collection = True

    model = ''
    input_file = resolve_file_path("test_data/inputs/mxnet/classification_imagenet.npz")
    h = 0  # input image height
    w = 0  # input image width
    preproc = {}  # input preprocessings (i.e. mean=(1, 2, 3))
    postproc = parse_classification()  # postprocessor
    model_env_key = "models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        model_path = prepend_with_env_path(self.model_env_key, self.model)
        self.__pytest_marks__ += (mark("classification", is_simple_mark=True),
                                  mark("mxnet", is_simple_mark=True))

        self.ref_collection = collect_mxnet_refs_pipeline(model=model_path,
                                                          input=self.input_file, h=self.h, w=self.w,
                                                          preprocessors=self.preproc)
        self.ref_pipeline = read_mxnet_refs_pipeline(ref_file=self.model, batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_mxnet(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=model_path,
                                 precision=precision,
                                 input_shape=(1, 3, self.h, self.w),
                                 legacy_mxnet_model=True),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = classification_comparators(postproc=self.postproc, precision=precision, device=device)
