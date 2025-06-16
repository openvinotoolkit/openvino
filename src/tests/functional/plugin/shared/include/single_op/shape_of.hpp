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
typedef std::tuple<
        ov::element::Type,              // Model type
        ov::element::Type,              // Output type
        std::vector<InputShape>,        // Input shapes
        ov::test::TargetDevice          // Device name
> shapeOfParams;

class ShapeOfLayerTest : public testing::WithParamInterface<shapeOfParams>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ParamType> obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
