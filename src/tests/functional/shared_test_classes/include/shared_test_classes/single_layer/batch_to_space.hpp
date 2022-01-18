// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

typedef std::tuple<
    std::vector<InputShape>,           // input shapes
    std::vector<int64_t>,              // block shape
    std::vector<int64_t>,              // crops begin
    std::vector<int64_t>,              // crops end
    ElementType,                       // Network precision
    ElementType,                       // Input precision
    ElementType,                       // Output precision
    TargetDevice                       // Device name
> BatchToSpaceTestParams;

class BatchToSpaceLayerTest : public testing::WithParamInterface<BatchToSpaceTestParams>,
                              virtual public SubgraphBaseTest {
protected:
    void SetUp() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchToSpaceTestParams>& obj);
};
} // namespace subgraph
} // namespace test
} // namespace ov
