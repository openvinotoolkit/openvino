// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
    std::vector<float>,  // widths
    std::vector<float>,  // heights
    bool,                // clip
    float,               // step_width
    float,               // step_height
    float,               // step
    float,               // offset
    std::vector<float>> priorBoxClusteredSpecificParams;

typedef std::tuple<
    priorBoxClusteredSpecificParams,
    ov::element::Type,        // Model type
    std::vector<InputShape>,  // Input shape
    std::string> priorBoxClusteredLayerParams;

class PriorBoxClusteredLayerTest : public testing::WithParamInterface<priorBoxClusteredLayerParams>,
                                   virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
