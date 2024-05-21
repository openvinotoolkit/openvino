// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<
        element::Type_t,   // Data element type
        ov::AnyMap         // Additional configuration
> UndefinedEtCpuParams;

class UndefinedEtSubgraphTest : public testing::WithParamInterface<UndefinedEtCpuParams>,
                                public CPUTestUtils::CPUTestsBase,
                                virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<UndefinedEtCpuParams>& obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

    hint::ExecutionMode m_mode;
    element::Type m_data_et = element::undefined;
};

}  // namespace test
}  // namespace ov
