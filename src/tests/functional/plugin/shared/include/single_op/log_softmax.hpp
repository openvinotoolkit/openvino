// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using logSoftmaxLayerTestParams = std::tuple<
        ov::element::Type,                  // Model type
        std::vector<InputShape>,            // Input shapes
        int64_t,                            // Axis
        std::string                         // Target device
>;

class LogSoftmaxLayerTest : public testing::WithParamInterface<logSoftmaxLayerTestParams>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<logSoftmaxLayerTestParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
