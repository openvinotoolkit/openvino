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
using priorBoxSpecificParams =  std::tuple<
        std::vector<float>, // min_size
        std::vector<float>, // max_size
        std::vector<float>, // aspect_ratio
        std::vector<float>, // density
        std::vector<float>, // fixed_ratio
        std::vector<float>, // fixed_size
        bool,               // clip
        bool,               // flip
        float,              // step
        float,              // offset
        std::vector<float>, // variance
        bool,               // scale_all_sizes
        bool>;              // min_max_aspect_ratios_order

typedef std::tuple<
        priorBoxSpecificParams,
        ov::element::Type,        // model type
        std::vector<InputShape>,  // input shape
        std::string> priorBoxLayerParams;

class PriorBoxLayerTest
    : public testing::WithParamInterface<priorBoxLayerParams>,
      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<priorBoxLayerParams>& obj);
protected:
    void SetUp() override;
};

} // namespace test
} // namespace ov
