// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/group_normalization_fusion.hpp"

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;
using namespace testing;

namespace {

std::vector<GroupNormalizationFusionTestBaseValues> valid_vals = {
    std::make_tuple(ov::PartialShape{1, 320}, ov::Shape{}, ov::Shape{}, ov::Shape{320}, ov::Shape{320}, 1, 1e-5),
    std::make_tuple(ov::PartialShape{1, 320, 2, 2},
                    ov::Shape{1, 1, 1},
                    ov::Shape{1, 1, 1},
                    ov::Shape{320, 1, 1},
                    ov::Shape{1, 320, 1, 1},
                    1,
                    1e-5),
    std::make_tuple(ov::PartialShape{5, 320, 2, 2, 2},
                    ov::Shape{1, 320, 1},
                    ov::Shape{1, 320, 1},
                    ov::Shape{320, 1, 1, 1},
                    ov::Shape{320, 1, 1, 1},
                    320,
                    1e-5),
    std::make_tuple(ov::PartialShape{3, 320},
                    ov::Shape{32, 1},
                    ov::Shape{32, 1},
                    ov::Shape{320},
                    ov::Shape{320},
                    32,
                    1e-5),
    std::make_tuple(ov::PartialShape{2, 9, 4, 5, 6},
                    ov::Shape{3, 1},
                    ov::Shape{3, 1},
                    ov::Shape{1, 9, 1, 1, 1},
                    ov::Shape{1, 9, 1, 1, 1},
                    3,
                    1e-5),
    std::make_tuple(ov::PartialShape{1, 320, 2, 4},
                    ov::Shape{1, 32, 1},
                    ov::Shape{1, 32, 1},
                    ov::Shape{320, 1, 1},
                    ov::Shape{320, 1, 1},
                    32,
                    1e-5),
    std::make_tuple(ov::PartialShape{8, 320, 4, 8},
                    ov::Shape{},
                    ov::Shape{},
                    ov::Shape{320, 1, 1},
                    ov::Shape{1, 320, 1, 1},
                    32,
                    1e-5),
    std::make_tuple(ov::PartialShape{1, 512, 4, 8},
                    ov::Shape{},
                    ov::Shape{1, 128, 1},
                    ov::Shape{1, 512, 1, 1},
                    ov::Shape{512, 1, 1},
                    128,
                    1e-6),
    std::make_tuple(ov::PartialShape{1, 192, 2, 2},
                    ov::Shape{1, 64, 1},
                    ov::Shape{},
                    ov::Shape{1, 192, 1, 1},
                    ov::Shape{1, 192, 1, 1},
                    64,
                    1e-6)};

using GroupNormalizationFusionSubgraphTestAdditionalValues =
    std::tuple<ov::element::Type,  // input/output tensor element type
               std::string,        // taget device name
               ov::AnyMap>;        // taget device properties

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionSubgraphTests_f32,
    GroupNormalizationFusionSubgraphTestsF,
    ValuesIn(expand_vals(
        valid_vals,
        GroupNormalizationFusionSubgraphTestAdditionalValues(ov::element::f32, ov::test::utils::DEVICE_GPU, {}))),
    GroupNormalizationFusionSubgraphTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    GroupNormalizationFusionSubgraphTests_f16,
    GroupNormalizationFusionSubgraphTestsF,
    ValuesIn(expand_vals(
        valid_vals,
        GroupNormalizationFusionSubgraphTestAdditionalValues(ov::element::f16, ov::test::utils::DEVICE_GPU, {}))),
    GroupNormalizationFusionSubgraphTestsF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupNormalizationFusionSubgraphTests_bf16,
                         GroupNormalizationFusionSubgraphTestsF,
                         ValuesIn(expand_vals(valid_vals,
                                              GroupNormalizationFusionSubgraphTestAdditionalValues(
                                                  ov::element::bf16,
                                                  ov::test::utils::DEVICE_GPU,
                                                  {{ov::hint::inference_precision(ov::element::f16)}}))),
                         GroupNormalizationFusionSubgraphTestsF::getTestCaseName);

}  // namespace
