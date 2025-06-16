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
typedef std::tuple<std::vector<size_t>,      // Num splits
                   int64_t,                  // Axis
                   ov::element::Type,        // Model type
                   std::vector<InputShape>,  // Input shapes
                   std::string               // Target device name
                   >
    VariadicSplitParams;

class VariadicSplitLayerTest : public testing::WithParamInterface<VariadicSplitParams>,
                               virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<VariadicSplitParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
