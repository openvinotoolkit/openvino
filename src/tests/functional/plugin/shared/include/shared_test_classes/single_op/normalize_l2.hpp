// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using NormalizeL2LayerTestParams = std::tuple<
        std::vector<int64_t>,       // axes
        float,                      // eps
        ov::op::EpsMode,            // eps mode
        std::vector<InputShape>,    // input shape
        ov::element::Type,          // model type
        std::string                 // target device
>;

class NormalizeL2LayerTest : public testing::WithParamInterface<NormalizeL2LayerTestParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NormalizeL2LayerTestParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
