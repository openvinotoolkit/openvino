# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

from e2e_tests.common.common.common_base_class import CommonConfig
from e2e_tests.pipelines.pipeline_templates.comparators_template import dummy_comparators
from e2e_tests.pipelines.pipeline_templates.infer_templates import common_infer_step
from e2e_tests.pipelines.pipeline_templates.input_templates import read_npz_input
from e2e_tests.pipelines.pipeline_templates.ir_gen_templates import common_ir_generation
from e2e_tests.pipelines.pipeline_templates.preproc_templates import assemble_preproc
from e2e_tests.test_utils.path_utils import prepend_with_env_path, resolve_file_path
from e2e_tests.common.pytest_utils import mark


class IE_Infer_Only_Base(CommonConfig):
    input_file = resolve_file_path("test_data/inputs/caffe/classification_imagenet.npz")
    additional_mo_args = {}

    align_results = None

    def __init__(self, batch, device, precision, **kwargs):
        self.__pytest_marks__ += tuple([mark("no_comparison", is_simple_mark=True)])
        model_path = prepend_with_env_path(self.model_env_key, self.model)
        self.ref_pipeline = {}
        self.ie_pipeline = OrderedDict([
            read_npz_input(path=self.input_file),
            assemble_preproc(h=self.h, w=self.w, batch=batch, rename_inputs=[("data", self.input_name)],
                             permute_order=(2, 0, 1)),
            common_ir_generation(mo_out=self.environment["mo_out"], model=model_path, precision=precision,
                                 **self.additional_mo_args),
            common_infer_step(device=device, batch=batch, **kwargs)
        ])
        self.comparators = dummy_comparators()
