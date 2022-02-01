// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <exec_graph_info.hpp>

namespace {

using namespace ngraph;

std::string get_layer_type(const std::shared_ptr<ngraph::Node>& node) {
    const auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(ExecGraphInfoSerialization::LAYER_TYPE);
    IE_ASSERT(it != rt_info.end());
    return it->second.as<std::string>();
}

class RedundantReordersTest : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        auto type = element::f32;
        auto input = std::make_shared<opset8::Parameter>(type, Shape{1, 10});
        auto axis = opset8::Constant::create(element::i32, Shape{}, {0});
        auto squeeze = std::make_shared<opset8::Squeeze>(input, axis);
        auto constant = opset8::Constant::create(type, Shape{}, {2});
        auto mul = std::make_shared<opset8::Multiply>(squeeze, constant);
        auto mul2 = std::make_shared<opset8::Multiply>(squeeze, constant);
        auto mul3 = std::make_shared<opset8::Multiply>(squeeze, constant);
        auto concat = std::make_shared<opset8::Concat>(OutputVector{squeeze, squeeze}, 0);
        auto concat2 = std::make_shared<opset8::Concat>(OutputVector{mul, mul2, mul3, concat}, 0);
        function = std::make_shared<ngraph::Function>(concat2, ParameterVector{input});
    }

    void TearDown() override {
        auto f = executableNetwork.GetExecGraphInfo().getFunction();
        std::shared_ptr<Node> reorder = nullptr;
        // check if multiply nodes have the same reorder as input
        for (const auto& n : f->get_ordered_ops()) {
            auto layer_type = get_layer_type(n);
            if (layer_type == "Subgraph") {
                auto input = n->get_input_node_shared_ptr(0);
                ASSERT_EQ("Reorder", get_layer_type(input));
                if (!reorder) {
                    reorder = input;
                    continue;
                } else {
                    ASSERT_EQ(reorder, input);
                }
            } else if (layer_type == "Concat") {
                for (size_t i = 0; i < n->get_input_size(); i++)
                    ASSERT_NE(reorder, n->get_input_node_shared_ptr(i));
            }
        }
    }
};

TEST_F(RedundantReordersTest, smoke_RedundantReordersTest) {
    Run();
}

}  // namespace
