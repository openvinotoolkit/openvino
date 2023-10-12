from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline, \
    collect_caffe_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from utils.path_utils import ref_from_model, prepend_with_env_path, resolve_file_path
from utils.pytest_utils import mark

common_mean_file = resolve_file_path("test_data/imagenet_mean.npz")
common_input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")


class CAFFE_ClassificationNet(CommonConfig):
    """Base class for Caffe classification nets."""
    ref_collection = True
    model = ''  # model path
    h = 0  # input image height
    w = 0  # input image width
    preproc = {}  # input preprocessings (i.e. mean=(1, 2, 3))
    postproc = None  # postprocessor
    model_env_key = "models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += (mark("classification", is_simple_mark=True),
                                  mark("caffe", is_simple_mark=True))

        self.ref_collection = collect_caffe_refs_pipeline(model=prepend_with_env_path(self.model_env_key, self.model),
                                                          input=common_input_file, h=self.h, w=self.w,
                                                          preprocessors=self.preproc)
        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="caffe"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=common_input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = classification_comparators(postproc=self.postproc, precision=precision, device=device)
