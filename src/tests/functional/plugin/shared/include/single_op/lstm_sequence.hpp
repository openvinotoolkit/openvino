// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using LSTMSequenceParams = typename std::tuple<
        ov::test::utils::SequenceTestsMode,       // pure Sequence or TensorIterator
        size_t,                                   // seq_lengths
        size_t,                                   // batch
        size_t,                                   // hidden size
        size_t,                                   // input size
        std::vector<std::string>,                 // activations
        float,                                    // clip
        ov::op::RecurrentSequenceDirection,       // direction
        ov::test::utils::InputLayerType,          // WRB input type (Constant or Parameter)
        ov::element::Type,                        // Network precision
        std::string>;                             // Device name


class LSTMSequenceTest : public testing::WithParamInterface<LSTMSequenceParams>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams> &obj);
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;

private:
    size_t max_seq_lengths;
};
}  // namespace test
}  // namespace ov
