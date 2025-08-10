// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using ReduceEltwiseParamsTuple = typename std::tuple<
        ov::Shape,                  // Input shapes
        std::vector<int>,           // Axis to reduce order
        ov::test::utils::OpType,    // Scalar or vector type axis
        bool,                       // Keep dims
        ov::element::Type,          // Network precision
        std::string>;               // Device name

class ReduceEltwiseTest: public testing::WithParamInterface<ReduceEltwiseParamsTuple>,
                         virtual public ov::test::SubgraphBaseStaticTest{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReduceEltwiseParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
