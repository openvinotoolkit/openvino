// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<std::string,  // test type
                   ov::Tensor,   // input
                   bool,         // adjoint
                   std::string   // device_name
                   >
    InverseTestParams;

class InverseLayerTest : public testing::WithParamInterface<InverseTestParams>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InverseTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;

private:
    ov::Tensor m_input;
};
}  // namespace test
}  // namespace ov
