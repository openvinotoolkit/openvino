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
using TensorIteratorParams = typename std::tuple<
        bool,                                     // using unroll tensor iterator transformation
        size_t,                                   // seq_lengths
        size_t,                                   // batch
        size_t,                                   // hidden size
        // todo: fix. input size hardcoded to 10 due to limitation (10 args) of gtests Combine() func.
        //size_t,                                 // input size
        size_t,                                   // sequence axis
        float,                                    // clip
        ov::test::utils::TensorIteratorBody,      // body type
        ov::op::RecurrentSequenceDirection,       // direction
        ov::element::Type,                        // Model type
        ov::test::TargetDevice>;                  // Device name

class TensorIteratorTest : public testing::WithParamInterface<TensorIteratorParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorParams> &obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};
}  // namespace test
}  // namespace ov
