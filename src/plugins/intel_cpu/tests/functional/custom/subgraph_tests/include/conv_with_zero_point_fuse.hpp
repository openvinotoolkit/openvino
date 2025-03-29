// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

using convConcatCPUParams = std::tuple<nodeType,  // Node convolution type
                                       ov::Shape  // Input shapes
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
                                          virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj);

protected:
    void SetUp() override;
    std::string pluginTypeNode;
};

}  // namespace test
}  // namespace ov
