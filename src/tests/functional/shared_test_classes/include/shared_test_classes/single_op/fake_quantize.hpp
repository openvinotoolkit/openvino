// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        size_t,                     // fake quantize levels
        std::vector<size_t>,        // fake quantize inputs shape
        std::vector<float>,         // fake quantize (inputLow, inputHigh, outputLow, outputHigh) or empty for random
        ov::op::AutoBroadcastSpec   // fake quantize broadcast mode
> fqSpecificParams;
typedef std::tuple<
        fqSpecificParams,
        ov::element::Type,                                         // Model type
        std::vector<InputShape>,                                   // Input shapes
        std::string                                                // Device name
> fqLayerTestParamsSet;

class FakeQuantizeLayerTest : public testing::WithParamInterface<fqLayerTestParamsSet>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<fqLayerTestParamsSet>& obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
