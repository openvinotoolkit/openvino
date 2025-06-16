// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using axisUpdateShapeInShape = std::tuple<
        std::vector<InputShape>,           // input, update shapes
        ov::Shape,                         // indices shape
        int64_t>;                          // axis
using scatterUpdateParamsTuple = typename std::tuple<
        axisUpdateShapeInShape,
        std::vector<int64_t>,              // Indices value
        ov::element::Type,                 // Model type
        ov::element::Type,                 // Indices type
        ov::test::TargetDevice             // Device name
>;

class ScatterUpdateLayerTest : public testing::WithParamInterface<scatterUpdateParamsTuple>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterUpdateParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
