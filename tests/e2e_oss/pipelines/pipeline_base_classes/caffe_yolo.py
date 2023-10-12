from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline
from e2e_oss.pipelines.pipeline_templates.comparators_template import object_detection_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.preproc_templates import assemble_preproc_caffe
from utils.path_utils import ref_from_model, prepend_with_env_path, resolve_file_path
from utils.pytest_utils import mark

common_input_file = resolve_file_path("test_data/inputs/caffe/car_1_data.npz")


class CAFFE_YoloBase(CommonConfig):
    def __init__(self, *args, **kwargs):
        self.__pytest_marks__ += (mark("object_detection", is_simple_mark=True),
                                  mark("od", is_simple_mark=True),
                                  mark("yolo", is_simple_mark=True),
                                  mark("caffe", is_simple_mark=True))


class CAFFE_YOLO_V1_Base(CAFFE_YoloBase):
    postproc = OrderedDict([
        ("parse_yolo_V1_region", {"classes": 20, "coords": 4, "num": 2, "grid": (7, 7)}),
        ("prob_filter", {"threshold": 0.3}),  # probability threshold for probability filter
        ("nms", {"overlap_threshold": 0.4}),  # IOU threshold in NMS filter
        ("clip_boxes", {"normalized_boxes": True})])

    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()
        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="caffe"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(common_input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
        ])
        self.comparators = object_detection_comparators(postproc=self.postproc,
                                                        device=device, precision=precision)


class CAFFE_YOLO_V2_Base(CAFFE_YoloBase):
    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()
        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="caffe"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(common_input_file),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
        ])
        self.comparators = object_detection_comparators(postproc=self.postproc, precision=precision,
                                                        target_layers=["yolo_v2_parsed"], device=device)


class CAFFE_YOLO_V3_Base(CAFFE_YoloBase):
    def __init__(self, batch, device, precision, api_2, **kwargs):
        super().__init__()
        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.model, framework="caffe"),
                                               batch=batch)
        self.ie_pipeline = OrderedDict([
            read_npz_input(resolve_file_path("test_data/inputs/caffe/person_detection.npz")),
            assemble_preproc_caffe(batch=batch, h=self.h, w=self.w, **self.preproc),
            common_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                                 model=prepend_with_env_path(self.model_env_key, self.model),
                                 precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs),
        ])
        self.comparators = object_detection_comparators(postproc=self.postproc, precision=precision,
                                                        target_layers=["yolo_v3_parsed"], device=device)
