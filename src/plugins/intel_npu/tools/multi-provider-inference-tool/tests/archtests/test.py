#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#


import unittest
import pathlib
from pytestarch import get_evaluable_architecture, Rule

class ArchTests(unittest.TestCase):

    def setUp(self):
        project_dir = pathlib.Path(__file__).parent.resolve().parent.parent.absolute()
        self.evaluable = get_evaluable_architecture(project_dir, project_dir, exclusions=["*.git*", '*__pycache__*'], exclude_external_libraries=False)

    def test_no_coupling_with_provider_implementations(self) -> None:
        no_coupling_with_provider_implementations: Rule = (
            Rule()
            .modules_that()
                .have_name_matching(r"^((providers).*)|((.*\.providers).*)")
            .should_not()
            .be_imported_by_anything()
        )
        no_coupling_with_provider_implementations.assert_applies(self.evaluable)

    def test_no_openvino_in_onnx_provider(self):
        rule = (
            Rule()
            .modules_that()
            .are_sub_modules_of("multi-provider-inference-tool.providers.onnx")
            .should_not()
            .import_modules_that()
            .are_named(["openvino"])
        )
        rule.assert_applies(self.evaluable)

    def test_no_onnx_in_ov_provider(self):
        rule = (
            Rule()
            .modules_that()
            .are_sub_modules_of("multi-provider-inference-tool.providers.ov")
            .should_not()
            .import_modules_that()
            .have_name_matching(r".*onnx.*")
        )
        rule.assert_applies(self.evaluable)

    def test_low_level_do_not_depend_on_high_level(self):
        rule = (
            Rule()
            .modules_that()
            .have_name_matching([r"^mpit.*", r"^((providers).*)|((.*\.providers).*)"])
            .should_not()
            .be_imported_by_modules_that()
            .are_named(["multi-provider-inference-tool.common", "multi-provider-inference-tool.params" ])
        )
        rule.assert_applies(self.evaluable)

    def test_no_coupling_between_low_levels(self):
        pure_common_rule = (
            Rule()
            .modules_that()
            .have_name_matching([r"^params.*"])
            .should_not()
            .be_imported_by_modules_that()
            .are_named("multi-provider-inference-tool.common")
        )
        pure_common_rule.assert_applies(self.evaluable)

        pure_params_rule = (
            Rule()
            .modules_that()
            .have_name_matching([r"^common.*"])
            .should_not()
            .be_imported_by_modules_that()
            .are_named("multi-provider-inference-tool.params")
        )
        pure_params_rule.assert_applies(self.evaluable)

    def test_utils_purity(self):
        pure_utils_rule = (
            Rule()
            .modules_that()
            .have_name_matching([r"^common.*", r"^params.*"])
            .should_not()
            .be_imported_by_modules_that()
            .are_named("multi-provider-inference-tool.utils")
        )
        pure_utils_rule.assert_applies(self.evaluable)


if __name__ == '__main__':
    unittest.main()
