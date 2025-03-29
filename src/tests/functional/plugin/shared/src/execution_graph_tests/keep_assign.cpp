// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/keep_assign.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "openvino/runtime/core.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/constant.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphKeepAssignNode::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string targetDevice = obj.param;
    return "Dev=" + targetDevice;
}

void ExecGraphKeepAssignNode::SetUp() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
}

/**
 * Assign/MemoryOutput operation node may hanging in air (leaf, has no consumer).
 * So exec graph may lose it. Will check that it's present in dumped exec graph.
 */
TEST_P(ExecGraphKeepAssignNode, KeepAssignNode) {
    auto device_name = this->GetParam();
    ov::Shape shape = {3, 2};
    ov::element::Type type = ov::element::f32;

    // Some simple graph with Memory(Assign) node                           //    in   read     //
    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);      //    | \  /        //
    auto mem_i = std::make_shared<ov::op::v0::Constant>(type, shape, 0);    //    |  mul        //
    auto mem_r = std::make_shared<ov::op::v3::ReadValue>(mem_i, "id");      //    | /  \        //
    auto mul   = std::make_shared<ov::op::v1::Multiply>(mem_r, input);      //    sum  assign   //
    auto mem_w = std::make_shared<ov::op::v3::Assign>(mul, "id");           //     |            //
    auto sum   = std::make_shared<ov::op::v1::Add>(mul, input);             //    out           //

    mem_w->add_control_dependency(mem_r);
    sum->add_control_dependency(mem_w);

    auto model = std::make_shared<ov::Model>(
        ov::NodeVector      {sum},
        ov::ParameterVector {input},
        "SimpleNet");

    // Load into plugin and get exec graph
    auto core  = ov::Core();
    auto compiled_model = core.compile_model(model, device_name);
    auto runtime_model  = compiled_model.get_runtime_model();
    auto runtime_ops    = runtime_model->get_ops();

    // Check Memory(Assign) node existence
    bool assign_node_found;
    for (auto &node : runtime_ops) {
        auto var = node->get_rt_info()["layerType"];
        auto s_val = var.as<std::string>();

        if (s_val == "MemoryOutput") {
            assign_node_found = true;
            break;
        }
    }
    ASSERT_TRUE(assign_node_found);
}

}  // namespace ExecutionGraphTests
