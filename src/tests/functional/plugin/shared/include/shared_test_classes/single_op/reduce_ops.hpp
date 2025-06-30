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
using reduceOpsParams = std::tuple<
    std::vector<int>,               // Axis to reduce order
    ov::test::utils::OpType,        // Scalar or vector type axis
    bool,                           // Keep dims
    ov::test::utils::ReductionType, // Reduce operation type
    ov::element::Type,              // Model type
    std::vector<size_t>,            // Input shape
    ov::test::TargetDevice          // Device name
>;

class ReduceOpsLayerTest : public testing::WithParamInterface<reduceOpsParams>,
                           virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reduceOpsParams>& obj);

protected:
    void SetUp() override;
};

class ReduceOpsLayerWithSpecificInputTest : public ReduceOpsLayerTest {
protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};
}  // namespace test
}  // namespace ov
