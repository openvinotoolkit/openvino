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
using ShapeAxesTuple = std::pair<std::vector<InputShape>, std::vector<int>>;

typedef std::tuple<
        ShapeAxesTuple,                 // InputShape (required), Squeeze indexes (if empty treated as non-existent)
        ov::test::utils::SqueezeOpType, // Op type
        ov::element::Type,              // Model type
        ov::test::TargetDevice          // Target device name
> squeezeParams;

class SqueezeUnsqueezeLayerTest : public testing::WithParamInterface<squeezeParams>,
                       virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<squeezeParams>& obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
