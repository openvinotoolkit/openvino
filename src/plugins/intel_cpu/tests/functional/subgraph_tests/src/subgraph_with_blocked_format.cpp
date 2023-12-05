// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"
#include <ngraph/opsets/opset8.hpp>

using namespace ngraph;

namespace SubgraphTestsDefinitions {

class SubgraphWithBlockedFormat : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        auto type = element::f32;
        auto param = std::make_shared<opset8::Parameter>(type, Shape{1, 32, 64, 32});
        auto weights = builder::makeConstant(type, Shape{32, 32, 1, 1}, std::vector<float>{}, true);
        auto conv = std::make_shared<opset8::Convolution>(param, weights, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
        auto mean = std::make_shared<opset8::ReduceMean>(conv, opset8::Constant::create(element::i32, Shape{2}, {2, 3}), true);
        auto reshape_before = std::make_shared<ov::op::v1::Reshape>(mean, opset8::Constant::create(element::i32, Shape{3}, {0, 16, -1}), true);
        auto mvn = std::make_shared<opset8::MVN>(reshape_before, opset8::Constant::create(element::i32, Shape{1}, {2}),
                false, 0.1, op::MVNEpsMode::INSIDE_SQRT);
        auto reshape_after = std::make_shared<ov::op::v1::Reshape>(mvn, std::make_shared<ov::op::v3::ShapeOf>(mean), false);
        auto mul = std::make_shared<opset8::Multiply>(reshape_after, builder::makeConstant(type, Shape{32, 1, 1}, std::vector<float>{}, true));
        auto add = std::make_shared<opset8::Add>(mul, builder::makeConstant(type, Shape{32, 1, 1}, std::vector<float>{}, true));
        auto sigmoid = std::make_shared<opset8::Sigmoid>(add);
        auto mul2 = std::make_shared<opset8::Multiply>(conv, sigmoid);

        function = std::make_shared<Function>(mul2, ParameterVector{param});
    }

    void TearDown() override {
        auto runtime_function = executableNetwork.GetExecGraphInfo().getFunction();
        int nodes_found = 0;
        for (const auto& n : runtime_function->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
            if (layer_type == "Subgraph") {
                nodes_found++;
                auto output_layout = n->get_rt_info().at(ExecGraphInfoSerialization::OUTPUT_LAYOUTS).as<std::string>();
                // convolution maybe chooses 'nhwc' and the subgraph will follow it
                ASSERT_TRUE(output_layout == "aBcd8b" || output_layout == "aBcd16b" || output_layout == "acdb");
            }
        }
        ASSERT_GT(nodes_found, 0);
    }
};

TEST_F(SubgraphWithBlockedFormat, smoke_CompareWithRefs) {
    Run();
}

} // namespace SubgraphTestsDefinitions
