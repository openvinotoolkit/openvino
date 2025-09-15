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
        self.project_name = project_dir.name
        self.evaluable = get_evaluable_architecture(project_dir, project_dir, exclusions=["*.git*", '*__pycache__*', '*venv', '*.venv*'], exclude_external_libraries=False)

    def test_no_coupling_with_provider_implementations(self) -> None:
        no_coupling_with_provider_implementations: Rule = (
            Rule()
            .modules_that()
                 .have_name_matching(self.project_name + r"\.(providers)\..+\.(.*)")
            .should_not()
                .be_imported_by_modules_except_modules_that()
                    .are_sub_modules_of(f"{self.project_name}.providers")
        )
        no_coupling_with_provider_implementations.assert_applies(self.evaluable)

    def test_no_openvino_package_in_onnx_provider(self):
        rule = (
            Rule()
            .modules_that()
                .are_sub_modules_of(f"{self.project_name}.providers.onnx")
            .should_not()
                .import_modules_that()
                    .are_named(["openvino"])
        )
        rule.assert_applies(self.evaluable)

    def test_no_openvino_modules_in_onnx_provider(self):
        rule = (
            Rule()
            .modules_that()
                .are_sub_modules_of(f"{self.project_name}.providers.onnx")
            .should_not()
                .import_modules_that()
                    .are_sub_modules_of(f"{self.project_name}.providers.ov")
        )
        rule.assert_applies(self.evaluable)

    def test_no_onnx_package_in_ov_provider(self):
        rule = (
            Rule()
            .modules_that()
                .are_sub_modules_of(f"{self.project_name}.providers.ov")
            .should_not()
                .import_modules_that()
                    .have_name_matching(r".*onnx.*")
        )
        rule.assert_applies(self.evaluable)

    def test_no_onnx_modules_in_ov_provider(self):
        rule = (
            Rule()
            .modules_that()
                .are_sub_modules_of(f"{self.project_name}.providers.ov")
            .should_not()
                .import_modules_that()
                    .are_sub_modules_of(f"{self.project_name}.providers.onnx")
        )
        rule.assert_applies(self.evaluable)

    def test_low_level_do_not_depend_on_high_level(self):
        rule = (
            Rule()
            .modules_that()
                .have_name_matching([r".*plugin_loader.*", r"^((providers).*)|((.*\.providers).*)"])
            .should_not()
                .be_imported_by_modules_that()
                    .have_name_matching([f"{self.project_name}.providers.interfaces",
                                         self.project_name + r"\.common\..*" ])
        )
        rule.assert_applies(self.evaluable)

    def test_no_coupling_between_low_levels(self):
        pure_common_rule = (
            Rule()
            .modules_that()
                .have_name_matching([self.project_name + r"\.common\..*"])
            .should_not()
                .be_imported_by_modules_that()
                    .are_named(f"{self.project_name}.providers.interfaces")
        )
        pure_common_rule.assert_applies(self.evaluable)

        pure_params_rule = (
            Rule()
            .modules_that()
                .have_name_matching([r".*interfaces.*"])
            .should_not()
                .be_imported_by_modules_that()
                    .have_name_matching(self.project_name + r"\.common\..*")
        )
        pure_params_rule.assert_applies(self.evaluable)

    def test_utils_purity(self):
        pure_utils_rule = (
            Rule()
            .modules_that()
                .have_name_matching([f"{self.project_name}.providers.interfaces.*",
                                     self.project_name + r"\.common\..*"])
            .should_not()
                .be_imported_by_modules_that()
                    .are_named(f"{self.project_name}.utils")
        )
        pure_utils_rule.assert_applies(self.evaluable)


if __name__ == '__main__':
    unittest.main()
