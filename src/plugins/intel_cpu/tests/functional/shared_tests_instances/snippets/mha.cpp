// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/mha.hpp"
#include "common_test_utils/test_constants.hpp"
#include "ie_plugin_config.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::vector<ov::PartialShape>> inputShapes = {
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 1, 1, 128}, {1, 128, 16, 64}},
        {{1, 128, 16, 64}, {1, 128, 16, 64}, {1, 16, 1, 1}, {1, 128, 16, 64}},
        {{2, 68, 6, 92}, {2, 68, 6, 92}, {1, 1, 68, 68}, {2, 68, 6, 92}},
        {{1, 58, 16, 34}, {1, 58, 16, 34}, {1, 1, 1, 58}, {1, 58, 16, 34}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHA,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShapes),
                             ::testing::ValuesIn({false, true}),
                             ::testing::Values(ov::element::f32),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(CommonTestUtils::DEVICE_CPU),
                             ::testing::Values(std::map<std::string, std::string>{})),
                     MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapeSelect = {
        // without broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 12, 128, 128}, {1, 128, 12, 64}},
        {{1, 94, 12, 54}, {1, 94, 12, 54}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 12, 94, 94}, {1, 94, 12, 54}},
        // with broadcast
        {{1, 128, 12, 64}, {1, 128, 12, 64}, {1, 12, 128, 128}, {1, 12, 1, 1}, {1, 12, 1, 1}, {1, 128, 12, 64}},
        {{2, 52, 6, 102}, {2, 52, 6, 102}, {1, 6, 52, 52}, {1, 6, 1, 1}, {1, 6, 1, 1}, {2, 52, 6, 102}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHA, MHASelect,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapeSelect),
                                 ::testing::Values(false),  // Need to support True for graph builder in tests
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(2), // Less + MHA
                                 ::testing::Values(2),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(std::map<std::string, std::string>{})),
                         MHA::getTestCaseName);

const std::vector<std::vector<ov::PartialShape>> inputShapesWOTranspose = {
        {{1, 12, 197, 64}, {1, 12, 64, 197}, {1, 12, 197, 64}},
        {{1, 12, 12, 64}, {1, 12, 64, 48}, {1, 12, 48, 64}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHAWOTransposeOnInputs, MHAWOTransposeOnInputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(ov::element::f32),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(std::map<std::string, std::string>{})),
                         MHA::getTestCaseName);

const std::map<std::string, std::string> cpuBF16PluginConfig = { { InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16,
                                                                   InferenceEngine::PluginConfigParams::YES } };

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_MHABF16, MHAWOTranspose,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapesWOTranspose),
                                 ::testing::ValuesIn({true}),  // Need to support False for graph builder in tests
                                 ::testing::Values(ov::element::bf16),
                                 ::testing::Values(3),
                                 ::testing::Values(0), // CPU plugin doesn't support MHA pattern via Snippets on bf16
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         MHA::getTestCaseName);


} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
