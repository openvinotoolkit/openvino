from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import read_refs_pipeline, get_refs_tf
from e2e_oss.pipelines.pipeline_templates.comparators_template import classification_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import read_img_input
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_oss.pipelines.pipeline_templates.postproc_template import parse_classification
from e2e_oss.utils.path_utils import ref_from_model


class TF_V2_ClassificationNet(CommonConfig):
    # Attributes bellow Have to be defined in inherited class
    h, w = 0, 0
    inputs_map = {}
    saved_model_dir = ""
    preprocess_mode = ""

    def __init__(self, batch, device, precision, api_2, **kwargs):
        def common_preprocess(custom_batch=None):
            if self.preprocess_mode == "caffe":
                return OrderedDict([
                    ("reverse_channels", {}),
                    ("resize", {"height": self.h, "width": self.w}),
                    ("subtract_mean_values", {"mean_values": [103.939, 116.779, 123.68]}),
                    ("normalize", {"factor": 127.5}),
                    ("align_with_batch", {"batch": custom_batch if custom_batch is not None else batch}),
                    ("cast_data_type", {"target_data_type": "float32"})
                ])

            elif self.preprocess_mode == "torch":
                return OrderedDict([
                    ("resize", {"height": self.h, "width": self.w}),
                    ("normalize", {"factor": [58.4, 57.12, 57.38]}),
                    ("subtract_mean_values", {"mean_values": [2.118, 2.035, 1.804]}),
                    ("align_with_batch", {"batch": custom_batch if custom_batch is not None else batch}),
                    ("cast_data_type", {"target_data_type": "float32"})
                ])

            elif self.preprocess_mode == "tf":
                return OrderedDict([
                    ("resize", {"height": self.h, "width": self.w}),
                    ("subtract_mean_values", {"mean_values": 127.5}),
                    ("normalize", {"factor": 127.5}),
                    ("align_with_batch", {"batch": custom_batch if custom_batch is not None else batch}),
                    ("cast_data_type", {"target_data_type": "float32"})
                ])
            else:
                raise ValueError(f"Pre-processing defined only for mode 'tf' or 'caffe' "
                                 f"but '{self.preprocess_mode}' was provided")

        self.ref_collection = {"pipeline": OrderedDict([
            read_img_input(path=self.inputs_map),
            ("preprocess", common_preprocess(custom_batch=1)),
            get_refs_tf(saved_model_dir=self.saved_model_dir, score_class_name="score_tf_v2")]),
            "store_path": ref_from_model(self.ref_name, framework="tf"),
            "store_path_for_ref_save": ref_from_model(self.ref_name, framework="tf", check_empty_ref_path=False)}

        self.ref_pipeline = read_refs_pipeline(ref_file=ref_from_model(model_name=self.ref_name, framework="tf"),
                                               batch=batch)

        preprocess = common_preprocess()
        if not api_2:
            preprocess.update({"permute_shape": {"order": (0, 3, 1, 2)}})
            preprocess.move_to_end("permute_shape")

        self.ie_pipeline = OrderedDict([
            read_img_input(path=self.inputs_map),
            ("preprocess", preprocess),
            common_ir_generation(mo_runner=self.environment["mo_runner"],
                                 mo_out=self.environment["mo_out"],
                                 model=self.saved_model_dir,
                                 precision=precision,
                                 input_shape=f"(1,{self.h},{self.w},3)"),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])

        self.comparators = classification_comparators(postproc=parse_classification(), precision=precision,
                                                      device=device)
