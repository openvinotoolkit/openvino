import os
from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import ir_pregenerated
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from e2e_oss._utils.path_utils import search_model_path_recursively, ref_from_model, resolve_file_path
from e2e_oss.common_utils.pytest_utils import mark

people_input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc_people.npz")
car_input_file = resolve_file_path("test_data/inputs/caffe/car_1.npz")


class Pregenerated_eltwise_base(CommonConfig):
    """
        Base class for E2E base classes. Used for pregenerated IR models.
        Provides comparison using element-wise comparator.

        :attr preproc:  dict-like entity that maps preprocessors to their attributes
                            (e. g. {'rename_inputs': [('data', '0')]})
        :attr input_file:    path to .npz file with input data
        :attr model_env_key:   special key which specifies model location.
                                           Existing keys are listed in e2e_oss/.automation/env_config.yml
        """
    preproc = {}
    input_file = people_input_file
    model_env_key = "models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += (mark("pregenerated", is_simple_mark=True),
                                  mark("downloader", is_simple_mark=True))

        self.ref_pipeline = read_refs_pipeline(ref_file=self.ref_file, batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            ir_pregenerated(xml=self.xml),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device)


class Pregenerated_eltwise_downloader(Pregenerated_eltwise_base):
    """
        Base class for E2E test classes. Used for pregenerated IR models from Model Downloader.
        """

    def __init__(self, batch, device, precision, api_2, **kwargs):
        assert self.model_env_key == "models", "This class is intended only for models from Model Downloader"
        self.ref_file = ref_from_model(model_name=self.model, framework="ie")
        self.xml = search_model_path_recursively(config_key="models",
                                                 model_name=os.path.join(precision, self.model + '.xml'))
        super().__init__(batch, device, precision, api_2, **kwargs)


class Pregenerated_eltwise_downloader_no_preproc(Pregenerated_eltwise_downloader):
    """
        Base class for E2E test classes. Used for pregenerated models from Model Downloader
        which don't need permute order and resize preprocessing.
        """
    preproc = {}
    input_file = people_input_file
    model_env_key = "models"

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__(batch, device, precision, api_2, **kwargs)
        self.ie_pipeline["preprocess"] = {'align_with_batch': {"batch": batch}, **self.preproc}
