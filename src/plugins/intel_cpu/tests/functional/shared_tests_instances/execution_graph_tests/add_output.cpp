// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_constants.hpp>
#include "execution_graph_tests/add_output.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/constant.hpp"

inline std::shared_ptr<ov::Model> getTargetNetwork() {
    auto shape = ov::Shape{1, 200};
    auto type = ov::element::f32;

    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
    auto mem_i = std::make_shared<ov::op::v0::Constant>(type, shape, 0);
    auto mem_r = std::make_shared<ov::op::v3::ReadValue>(mem_i, "id");
    auto mul   = std::make_shared<ov::op::v1::Multiply>(mem_r, input);
    auto mem_w = std::make_shared<ov::op::v3::Assign>(mul, "id");
    auto sigm = std::make_shared<ov::op::v0::Sigmoid>(mul);
    mem_r->output(0).set_names({"Memory"});
    mem_w->add_control_dependency(mem_r);
    sigm->add_control_dependency(mem_w);
    sigm->output(0).set_names({"Sigmoid"});
    return std::make_shared<ov::Model>(sigm->outputs(), ov::ParameterVector{input}, "addOutput");
}

std::vector<addOutputsParams> testCases = {
        addOutputsParams(getTargetNetwork(), {"Memory"}, ov::test::utils::DEVICE_CPU)
};

INSTANTIATE_TEST_SUITE_P(smoke_AddOutputBasic, AddOutputsTest,
        ::testing::ValuesIn(testCases),
        AddOutputsTest::getTestCaseName);
