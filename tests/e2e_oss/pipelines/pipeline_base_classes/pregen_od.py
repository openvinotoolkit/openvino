import os
from collections import OrderedDict

from tests.e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from tests.e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators
from tests.e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from tests.e2e_oss.pipelines.pipeline_templates.ir_gen_templates import ir_pregenerated
from tests.e2e_oss.pipelines.pipeline_templates.postproc_template import parse_object_detection
from tests.e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from tests.e2e_oss._utils.path_utils import search_model_path_recursively, ref_from_model, resolve_file_path
from tests.e2e_oss.common_utils.pytest_utils import mark

people_input_file = resolve_file_path("test_data/inputs/caffe/object_detection_voc_people.npz")


class Pregenerated_od_base(CommonConfig):
    """
        Base class for E2E base classes. Used for pregenerated models.
        Provides comparison using object detection comparator.

        :attr preproc:  dict-like entity that maps preprocessors to their attributes
                            (e. g. {'rename_inputs': [('data', '0')]})
        :attr input_file:    path to .npz file with input data
        :attr model_env_key:   special key which specifies model location.
                                   Existing keys are listed in e2e_oss/.automation/env_config.yml
        """
    preproc = {}
    input_file = people_input_file
    model_env_key = "models"
    align_results = None

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True),
                                  mark("od", is_simple_mark=True),
                                  mark("pregenerated", is_simple_mark=True),
                                  mark("downloader", is_simple_mark=True))

        self.ref_file = ref_from_model(model_name=self.model, framework="ie")

        self.ref_pipeline = OrderedDict([
            ("get_refs", {"precollected": {"path": self.ref_file}}),
            ("postprocess", {"align_with_batch_od": {"batch": batch}})])

        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            ir_pregenerated(xml=self.xml),
            ("infer", {"ie_sync": {"device": device,
                                   "network_modifiers": {"set_batch_using_reshape": {"batch": batch}},
                                   "cpu_extension": "cpu_extension"}}
             )
        ])
        self.comparators = object_detection_comparators(postproc=parse_object_detection(),
                                                        precision=precision, device=device)


class Pregenerated_od_downloader(Pregenerated_od_base):
    """
        Base class for E2E test classes. Used for pregenerated models from Model Downloader.
        """

    def __init__(self, batch, device, precision, api_2, **kwargs):
        assert self.model_env_key == "models", "This class is intended only for models from Model Downloader"
        self.xml = search_model_path_recursively(config_key="models",
                                                 model_name=os.path.join(precision, self.model + '.xml'))
        super().__init__(batch, device, precision, api_2, **kwargs)
