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
using GRUSequenceParams = typename std::tuple<
        ov::test::utils::SequenceTestsMode,       // pure Sequence or TensorIterator
        std::vector<InputShape>,                  // shapes
        std::vector<std::string>,                 // activations
        float,                                    // clip
        bool,                                     // linear_before_reset
        ov::op::RecurrentSequenceDirection,       // direction
        ov::test::utils::InputLayerType,          // WRB input type (Constant or Parameter)
        ov::element::Type,                        // Network precision
        std::string>;                             // Device name

class GRUSequenceTest : public testing::WithParamInterface<GRUSequenceParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GRUSequenceParams> &obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

private:
    size_t max_seq_lengths;
};
} //  namespace test
} //  namespace ov
