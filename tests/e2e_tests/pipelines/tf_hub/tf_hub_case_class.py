# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from e2e_tests.common.common.common_base_class import CommonConfig
from e2e_tests.pipelines.pipeline_templates.comparators_template import eltwise_comparators
from e2e_tests.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_tests.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation


class TFHUB_eltwise_Base(CommonConfig):
    def __init__(self, device, precision, **kwargs):
        self.model = {"load_model":
                          {"load_tf_hub_model":
                               {"model_name": self.model_name,
                                'model_link': self.model_link,
                                }}}
        self.input = {"read_input":
                          {"generate_tf_hub_inputs": {}}}

        self.ref_pipeline = {"get_refs_tf_hub":
                                 {'score_tf_hub': {}}}

        self.ie_pipeline = OrderedDict([
            common_ir_generation(mo_out=self.environment["mo_out"],
                                 precision=precision),
            common_infer_step(device=device, **kwargs)
        ])
        self.comparators = eltwise_comparators(precision=precision, device=device)
