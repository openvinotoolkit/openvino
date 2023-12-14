from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import get_refs_tf_hub
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.utils.test_utils import generate_tf_hub_inputs, get_tf_hub_model
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation


class TFHUB_eltwise_Base(CommonConfig):
    model = None
    inputs = None

    def prepare_prerequisites(self):
        self.model = get_tf_hub_model(self.model_name, self.model_link)
        self.inputs = generate_tf_hub_inputs(self.model)

    def __init__(self, device, precision, **kwargs):
        self.ref_pipeline = OrderedDict([
            get_refs_tf_hub(model=self.model, inputs=self.inputs)
        ])
        self.ie_pipeline = OrderedDict([
            common_ir_generation(mo_out=self.environment["mo_out"],
                                 model=self.model,
                                 precision=precision),
            common_infer_step(device=device, inputs=self.inputs, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device)
