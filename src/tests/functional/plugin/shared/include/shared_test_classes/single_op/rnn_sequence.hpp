// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using RNNSequenceParams = typename std::tuple<
    ov::test::utils::SequenceTestsMode,       // Pure Sequence or TensorIterator
    size_t,                                   // Sequence lengths
    size_t,                                   // Batch
    size_t,                                   // Hidden size
    size_t,                                   // Input size
    std::vector<std::string>,                 // Activations
    float,                                    // Clip
    ov::op::RecurrentSequenceDirection,       // Direction
    ov::test::utils::InputLayerType,          // WRB input type (Constant or Parameter)
    ov::element::Type,                        // Model type
    ov::test::TargetDevice                    // Device name
>;

class RNNSequenceTest : public testing::WithParamInterface<RNNSequenceParams>,
                     virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RNNSequenceParams> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
