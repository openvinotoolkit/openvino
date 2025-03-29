// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using TopKParams = std::tuple<int,      // Maximum number of boxes to be selected per class
                              int>;     // Maximum number of boxes to be selected per batch element

using ThresholdParams = std::tuple<float,   // minimum score to consider box for the processing
                                   float,   // gaussian_sigma parameter for gaussian decay_function
                                   float>;  // filter out boxes with low confidence score after decaying

using NmsParams = std::tuple<std::vector<InputShape>,                            // Params using to create 1st and 2nd inputs
                             ov::element::Type,                                  // Model type
                             ov::op::v8::MatrixNms::SortResultType,              // Order of output elements
                             ov::element::Type,                                  // Output type
                             TopKParams,                                         // Maximum number of boxes topk params
                             ThresholdParams,                                    // Thresholds: score_threshold, gaussian_sigma, post_threshold
                             int,                                                // Background class id
                             bool,                                               // If boxes are normalized
                             ov::op::v8::MatrixNms::DecayFunction,               // Decay function
                             std::string>;                                       // Device name

class MatrixNmsLayerTest : public testing::WithParamInterface<NmsParams>,
                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj);

protected:
    void SetUp() override;
};
} // namespace test
} // namespace ov
