// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<std::string,                      // test type
                   InputShape,                       // probs_shape
                   InputShape,                       // num_samples_shape
                   ov::test::ElementType,            // convert_type
                   bool,                             // with_replacement
                   bool,                             // log_probs
                   uint64_t,                         // global_seed
                   uint64_t,                         // op_seed
                   CPUTestUtils::CPUSpecificParams,  // CPU specific params
                   ov::AnyMap                        // Additional plugin configuration
                   >
    MultinomialTestCPUParams;

class MultinomialLayerTestCPU : public testing::WithParamInterface<MultinomialTestCPUParams>,
                                virtual public SubgraphBaseTest,
                                public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultinomialTestCPUParams>& obj);

protected:
    void SetUp() override;
};

}  // namespace CPULayerTestsDefinitions
