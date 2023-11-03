// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include <openvino/runtime/tensor.hpp>

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {

typedef std::tuple<std::string,                      // test type
                   ov::Tensor,                       // probs_shape
                   ov::Tensor,                       // num_samples_shape
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
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

private:
    ov::Tensor m_probs;
    ov::Tensor m_num_samples;
};

}  // namespace CPULayerTestsDefinitions
