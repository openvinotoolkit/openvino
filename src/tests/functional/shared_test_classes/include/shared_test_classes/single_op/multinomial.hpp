// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<std::string,            // test type
                   ov::Tensor,             // probs
                   ov::Tensor,             // num_samples
                   ov::test::ElementType,  // convert_type
                   bool,                   // with_replacement
                   bool,                   // log_probs
                   uint64_t,               // global_seed
                   uint64_t,               // op_seed
                   std::string             // device_name
                   >
    MultinomialTestParams;

class MultinomialLayerTest : public testing::WithParamInterface<MultinomialTestParams>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MultinomialTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;

private:
    ov::Tensor m_probs;
    ov::Tensor m_num_samples;
};
}  // namespace test
}  // namespace ov
