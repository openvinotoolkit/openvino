// Copyright (C) 2018-2023 Intel Corporation
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
using axisShapeInShape = std::tuple<
        std::vector<InputShape>,           // Input, update/indices shapes
        int                                // Axis
>;
using scatterElementsUpdateParamsTuple = typename std::tuple<
        axisShapeInShape,                  // Shape description
        std::vector<size_t>,               // Indices value
        ov::element::Type,                 // Model type
        ov::element::Type,                 // Indices type
        ov::test::TargetDevice             // Device name
>;

class ScatterElementsUpdateLayerTest : public testing::WithParamInterface<scatterElementsUpdateParamsTuple>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterElementsUpdateParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
