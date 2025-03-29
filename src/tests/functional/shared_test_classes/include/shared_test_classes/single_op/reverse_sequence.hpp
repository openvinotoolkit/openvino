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
using ReverseSequenceParamsTuple = typename std::tuple<
    int64_t,                           // Index of the batch dimension
    int64_t,                           // Index of the sequence dimension
    std::vector<size_t>,               // Input shapes
    std::vector<size_t>,               // Shape of the input vector with sequence lengths to be reversed
    ov::test::utils::InputLayerType,   // Secondary input type
    ov::element::Type,                 // Model type
    ov::test::TargetDevice             // Device name
>;

class ReverseSequenceLayerTest : public testing::WithParamInterface<ReverseSequenceParamsTuple>,
                        virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReverseSequenceParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
