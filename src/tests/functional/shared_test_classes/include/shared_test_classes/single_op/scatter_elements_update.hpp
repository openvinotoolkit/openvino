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

using scatterElementsUpdate12ParamsTuple = typename std::tuple<
        axisShapeInShape,                  // Shape description
        std::vector<int64_t>,               // Indices value
        ov::op::v12::ScatterElementsUpdate::Reduction,  // Reduce mode
        bool,                              // Use init value
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

class ScatterElementsUpdate12LayerTest : public testing::WithParamInterface<scatterElementsUpdate12ParamsTuple>,
                                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<scatterElementsUpdate12ParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
