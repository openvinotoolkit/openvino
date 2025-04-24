// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
    ov::element::Type,        // Model type
    std::vector<InputShape>,  // Input shapes
    float,                    // Bias
    std::string               // Device name
> grnParams;

class GrnLayerTest : public testing::WithParamInterface<grnParams>,
                     virtual public ov::test::SubgraphBaseTest{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<grnParams>& obj);
protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov
