// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/group_normalization_fusion.hpp"

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

namespace {

using GroupNormalizationFusionTransformationTestAdditionalValues =
    std::tuple<bool,         // whether it's a positive test that should run reference model or a negative test
               std::string,  // taget device name
               ov::AnyMap,   // taget device properties
               std::string,  // reference device name
               ov::AnyMap>;  // reference device properties

std::vector<GroupNormalizationFusionTestBaseValues> valid_vals = {
    std::make_tuple(ov::PartialShape{1, 320}, ov::Shape{}, ov::Shape{}, ov::Shape{320}, ov::Shape{320}, 1, 1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{1, 1, 1},
                    ov::Shape{1, 1, 1},
                    ov::Shape{320, 1, 1},
                    ov::Shape{1, 320, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(ov::PartialShape{5, 320, 2, 2, 2},
                    ov::Shape{1, 320, 1},
                    ov::Shape{1, 320, 1},
                    ov::Shape{320, 1, 1, 1},
                    ov::Shape{320, 1, 1, 1},
                    320,
                    1e-5f),
    std::make_tuple(ov::PartialShape{ov::Dimension::dynamic(),
                                     320,
                                     ov::Dimension::dynamic(),
                                     ov::Dimension::dynamic(),
                                     ov::Dimension::dynamic()},
                    ov::Shape{1, 320, 1},
                    ov::Shape{1, 320, 1},
                    ov::Shape{320, 1, 1, 1},
                    ov::Shape{320, 1, 1, 1},
                    320,
                    1e-5f),
    std::make_tuple(ov::PartialShape{3, 320},
                    ov::Shape{32, 1},
                    ov::Shape{32, 1},
                    ov::Shape{320},
                    ov::Shape{320},
                    32,
                    1e-5f),
    std::make_tuple(ov::PartialShape{2, 9, 4, 5, 6},
                    ov::Shape{3, 1},
                    ov::Shape{3, 1},
                    ov::Shape{1, 9, 1, 1, 1},
                    ov::Shape{1, 9, 1, 1, 1},
                    3,
                    1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 4},
                    ov::Shape{1, 32, 1},
                    ov::Shape{1, 32, 1},
                    ov::Shape{320, 1, 1},
                    ov::Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(ov::PartialShape{8, 320, 4, 8},
                    ov::Shape{},
                    ov::Shape{},
                    ov::Shape{320, 1, 1},
                    ov::Shape{1, 320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(ov::PartialShape{1, 512, 4, 8},
                    ov::Shape{},
                    ov::Shape{1, 128, 1},
                    ov::Shape{1, 512, 1, 1},
                    ov::Shape{512, 1, 1},
                    128,
                    1e-6f),
    std::make_tuple(ov::PartialShape{1, 192, 2, 2},
                    ov::Shape{1, 64, 1},
                    ov::Shape{},
                    ov::Shape{1, 192, 1, 1},
                    ov::Shape{1, 192, 1, 1},
                    64,
                    1e-6f)};

std::vector<GroupNormalizationFusionTestBaseValues> invalid_vals = {
    std::make_tuple(ov::PartialShape{1, 320}, ov::Shape{}, ov::Shape{}, ov::Shape{}, ov::Shape{}, 1, 1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{1, 1, 1},
                    ov::Shape{1, 1, 1},
                    ov::Shape{1, 1, 1},
                    ov::Shape{1, 1, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{},
                    ov::Shape{},
                    ov::Shape{320, 1, 1},
                    ov::Shape{},
                    1,
                    1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{},
                    ov::Shape{},
                    ov::Shape{},
                    ov::Shape{1, 320, 1, 1},
                    1,
                    1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{1, 1, 1},
                    ov::Shape{1, 32, 1},
                    ov::Shape{320, 1, 1},
                    ov::Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{1, 32, 1},
                    ov::Shape{1, 1, 1},
                    ov::Shape{320, 1, 1},
                    ov::Shape{320, 1, 1},
                    32,
                    1e-5f),
    std::make_tuple(ov::PartialShape{ov::Dimension::dynamic(), 512, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                    ov::Shape{},
                    ov::Shape{},
                    ov::Shape{1, 512, 1, 1},
                    ov::Shape{1, 512, 1, 1},
                    100,
                    1e-6f)};

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphPositiveTests_f32,
                         GroupNormalizationFusionSubgraphTestsF_f32,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  true,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphPositiveTests_f16,
                         GroupNormalizationFusionSubgraphTestsF_f16,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  true,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphPositiveTests_bf16,
                         GroupNormalizationFusionSubgraphTestsF_bf16,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  true,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {{ov::hint::inference_precision(ov::element::f16)}},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_bf16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTests_f32,
                         GroupNormalizationFusionSubgraphTestsF_f32,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTests_f16,
                         GroupNormalizationFusionSubgraphTestsF_f16,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTests_bf16,
                         GroupNormalizationFusionSubgraphTestsF_bf16,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {{ov::hint::inference_precision(ov::element::f16)}},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_bf16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_u8,
                         GroupNormalizationFusionSubgraphTestsF_u8,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_u16,
                         GroupNormalizationFusionSubgraphTestsF_u16,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_u32,
                         GroupNormalizationFusionSubgraphTestsF_u32,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_u64,
                         GroupNormalizationFusionSubgraphTestsF_u64,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u64::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_i8,
                         GroupNormalizationFusionSubgraphTestsF_i8,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_i8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_i16,
                         GroupNormalizationFusionSubgraphTestsF_i16,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_i16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_i32,
                         GroupNormalizationFusionSubgraphTestsF_i32,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_i32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_f8e5m2,
                         GroupNormalizationFusionSubgraphTestsF_f8e5m2,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f8e5m2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_f4e2m1,
                         GroupNormalizationFusionSubgraphTestsF_f4e2m1,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f4e2m1::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsValidVals_f8e8m0,
                         GroupNormalizationFusionSubgraphTestsF_f8e8m0,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f8e8m0::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_u8,
                         GroupNormalizationFusionSubgraphTestsF_u8,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_u16,
                         GroupNormalizationFusionSubgraphTestsF_u16,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_u32,
                         GroupNormalizationFusionSubgraphTestsF_u32,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_u64,
                         GroupNormalizationFusionSubgraphTestsF_u64,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_u64::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_i8,
                         GroupNormalizationFusionSubgraphTestsF_i8,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_i8::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_i16,
                         GroupNormalizationFusionSubgraphTestsF_i16,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_i16::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_i32,
                         GroupNormalizationFusionSubgraphTestsF_i32,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_i32::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInalidVals_f8e5m2,
                         GroupNormalizationFusionSubgraphTestsF_f8e5m2,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f8e5m2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_f4e2m1,
                         GroupNormalizationFusionSubgraphTestsF_f4e2m1,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f4e2m1::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphNegativeTestsInvalidVals_f8e8m0,
                         GroupNormalizationFusionSubgraphTestsF_f8e8m0,
                         ValuesIn(expand_vals(invalid_vals,
                                              GroupNormalizationFusionTransformationTestAdditionalValues(
                                                  false,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {},
                                                  ov::test::utils::DEVICE_TEMPLATE,
                                                  {{"DISABLE_TRANSFORMATIONS", true}}))),
                         GroupNormalizationFusionSubgraphTestsF_f8e8m0::getTestCaseName);

}  // namespace
