from collections import OrderedDict

from e2e_oss.pipelines.pipeline_base_classes.common_base_class import CommonConfig
from e2e_oss.pipelines.pipeline_templates.collect_reference_templates import get_refs_tf_hub
from e2e_oss.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_oss.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_oss.pipelines.pipeline_templates.input_templates import generate_tf_hub_inputs
from e2e_oss.pipelines.pipeline_templates.ir_gen_templates import ovc_ir_generation
from e2e_oss.pipelines.pipeline_templates.tf_hub_loader_template import tf_hub_loader


class TFHUB_eltwise_Base(CommonConfig):
    ref_collect_func = ''
    model_link = ''
    model_name = ''
    model = ''

    def __init__(self, batch, device, precision, api_2, **kwargs):
        self.ref_pipeline = OrderedDict([
            tf_hub_loader(model_name=self.model_name, model_link=self.model_link),
            generate_tf_hub_inputs(),
            ('preprocess_tf_hub', OrderedDict([
                ('assign_indices_tf_hub', {}),
                ('align_with_batch_tf_hub', {'batch': 1, 'expand_dims': False})
            ])),
            get_refs_tf_hub(),
            ("postprocess", {"align_with_batch": {"batch": batch}})])

        self.ie_pipeline = OrderedDict([
            tf_hub_loader(model_name=self.model_name, model_link=self.model_link),
            generate_tf_hub_inputs(),
            ('preprocess_tf_hub', OrderedDict([
                ('assign_indices_tf_hub', {}),
                ('align_with_batch_tf_hub', {'batch': 1, 'expand_dims': False})
            ])),
            ovc_ir_generation(mo_runner=self.environment["mo_runner"], mo_out=self.environment["mo_out"],
                              model=self.model,
                              precision=precision),
            common_infer_step(device=device, batch=batch, api_2=api_2, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device)
