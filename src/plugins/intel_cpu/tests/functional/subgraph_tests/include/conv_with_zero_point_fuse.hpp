// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

using convConcatCPUParams = std::tuple<
    nodeType,                           // Ngraph convolution type
    InferenceEngine::SizeVector         // Input shapes
>;

// Subgraph:
/*
 *           Paramter           Constant
 *               |                 | i8
 *               |                 |
 *         FakeQuantise         Convert
 *           /      \              | f32
 *          /        \             |
 *      MaxPool    FakeQuantize  Mulltiply
 *         \           \         /
 *          \           \       /
 *           \        Convolution
 *            \        /
 *             \      /
 *              Concat
 *                |
 *                |
 *             Result
 */

class ConvWithZeroPointFuseSubgraphTest : public testing::WithParamInterface<convConcatCPUParams>,
                                          public CPUTestsBase,
                                          virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj);

protected:
    void SetUp() override;
    std::string pluginTypeNode;
};

} // namespace SubgraphTestsDefinitions
