// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/keep_assign.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <inference_engine.hpp>

namespace ExecutionGraphTests {

std::string ExecGraphKeepAssignNode::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string targetDevice = obj.param;
    return "Dev=" + targetDevice;
}

/**
 * Assign/MemoryOutput operation node may hanging in air (leaf, has no consumer).
 * So exec graph may lose it. Will check that it's present in dumped exec graph.
 */
TEST_P(ExecGraphKeepAssignNode, KeepAssignNode) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto device_name = this->GetParam();
    ngraph::Shape shape = {3, 2};
    ngraph::element::Type type = ngraph::element::f32;

    using std::make_shared;
    using namespace ngraph::opset5;

    // Some simple graph with Memory(Assign) node                     //    in   read     //
    auto input = make_shared<Parameter>(type, shape);                 //    | \  /        //
    auto mem_i = make_shared<Constant>(type, shape, 0);               //    |  mul        //
    auto mem_r = make_shared<ReadValue>(mem_i, "id");                 //    | /  \        //
    auto mul   = make_shared<ngraph::op::v1::Multiply>(mem_r, input); //    sum  assign   //
    auto mem_w = make_shared<Assign>(mul, "id");                      //     |            //
    auto sum   = make_shared<ngraph::op::v1::Add>(mul, input);        //    out           //

    mem_w->add_control_dependency(mem_r);
    sum->add_control_dependency(mem_w);

    auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector      {sum},
            ngraph::ParameterVector {input},
            "SimpleNet");

    // Load into plugin and get exec graph
    auto ie  = InferenceEngine::Core();
    auto net = InferenceEngine::CNNNetwork(function);
    auto exec_net   = ie.LoadNetwork(net, device_name);
    auto exec_graph = exec_net.GetExecGraphInfo();
    auto exec_ops   = exec_graph.getFunction()->get_ops();

    // Check Memory(Assign) node existence
    bool assign_node_found;
    for (auto &node : exec_ops) {
        auto var = node->get_rt_info()["layerType"];
        auto s_val = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(var);

        if (s_val->get() == "MemoryOutput") {
            assign_node_found = true;
            break;
        }
    }
    ASSERT_TRUE(assign_node_found);
}

}  // namespace ExecutionGraphTests
