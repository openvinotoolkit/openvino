// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

typedef std::tuple<
        ov::Shape,                        // Input/Output shape
        ov::test::ElementType,            // Data precision
        bool,                             // Is input constant
        CPUTestUtils::CPUSpecificParams,  // CPU specific params
        ov::AnyMap                        // Additional plugin configuration
> IdentityLayerTestCPUParamSet;

class IdentityLayerTestCPU : public testing::WithParamInterface<IdentityLayerTestCPUParamSet>,
                                  public ov::test::SubgraphBaseTest, public CPUTestUtils::CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<IdentityLayerTestCPUParamSet>& obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;
};
}  // namespace test
}  // namespace ov
