// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/opsets/opset8.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

/*
    input1(f32_abcd_{1,64,32,32})                      input2(f16_abcd_{1,128,1,1})
        |                                                 |
    Reorder(f32_acdb_{1,64,32,32})   const             Convert(f32_abcd_{1,128,1,1})
        |                           /                     |
        |                          /                      |
    Convolution(f32_acdb_{1,1,30,30})  Range_1520      VariadicSplit(f32_abcd_{1,64,1,1}, f32_abcd_{1,64,1,1})
        |                             /                   \                               /
        |                           /                      \                             /
        |                         /                         \                           /
        |                       /                            \                         /
    MVN(f32_acdb_{1,1,30,30})              Reorder1(f32_acdb_{1,64,1,1})  Reorder2(f32_acdb_{1,64,1,1})
            \                             /                            /
             \                           /                            /
              \                         /                            /
               \                       /                            /
               Subgraph(f32_acdb_{1,64,30,30})
                            |
                            |
               Convolution(f32_acdb_{1,1,28,28})
                            |
                          Result
    The Subgraph node have 3 inputs: they don't have same layout.
    Expected: Reorder is inserted after VariadicSplit[0] and VariadicSplit[1], not inserted after MVN.
    Because VariadicSplit's output layout is [1,64,1,1], it' reorder almost have calculation.
*/

class SubgraphSelectPD : virtual public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        abs_threshold = 1e-2;

        auto type = element::f32;
#define CONST1 128
        auto input1 = std::make_shared<ov::opset8::Parameter>(type, Shape{1, CONST1 / 2, 32, 32});
        input1->set_friendly_name("input1");
        auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f16, Shape{1, CONST1, 1, 1});
        input2->set_friendly_name("input2");

        auto variadicSplit = std::make_shared<ov::op::v1::VariadicSplit>(
            input2,
            ov::opset8::Constant::create(element::i64, Shape{1}, {1}),
            ov::opset8::Constant::create(element::i64, Shape{2}, {CONST1 / 2, CONST1 / 2}));
        variadicSplit->set_friendly_name("variadicSplit");

        auto add1 = std::make_shared<ov::opset8::Add>(variadicSplit->output(0),
                                                      ov::opset8::Constant::create(element::f16, Shape{1}, {1}));
        add1->set_friendly_name("add1");
        auto convert_input1 = std::make_shared<ov::opset8::Convert>(input1, ov::element::f16);
        convert_input1->set_friendly_name("convert_input1");
        auto shapeof = std::make_shared<ov::opset8::ShapeOf>(convert_input1);
        auto rankof = std::make_shared<ov::opset8::ShapeOf>(shapeof);
        auto squeeze =
            std::make_shared<ov::opset8::Squeeze>(rankof, ov::opset8::Constant::create(element::i64, Shape{1}, {0}));

        auto range = std::make_shared<ov::opset8::Range>(ov::opset8::Constant::create(element::i64, Shape{}, {2}),
                                                         squeeze,
                                                         ov::opset8::Constant::create(element::i64, Shape{}, {1}),
                                                         ov::element::i64);
        auto create_conv = [](const std::shared_ptr<ov::Node>& input_node) {
            auto conv = std::make_shared<ov::opset8::Convolution>(
                input_node,
                ov::test::utils::deprecated::make_constant(ov::element::f16,
                                                           Shape{1, CONST1 / 2, 3, 3},
                                                           std::vector<float>{},
                                                           true,
                                                           0.1f,
                                                           0.9f),
                Strides{1, 1},
                CoordinateDiff{0, 0},
                CoordinateDiff{0, 0},
                Strides{1, 1});
            conv->get_rt_info() =
                CPUTestUtils::CPUTestsBase::makeCPUInfo({CPUTestUtils::nhwc}, {CPUTestUtils::nhwc}, {});
            return conv;
        };
        auto conv1 = create_conv(convert_input1);
        auto mvn = std::make_shared<ov::opset8::MVN>(conv1, range, false, 0.1, op::MVNEpsMode::INSIDE_SQRT);
        auto mul = std::make_shared<ov::opset8::Multiply>(add1, mvn);
        auto add2 = std::make_shared<ov::opset8::Add>(variadicSplit->output(1), mul);
        auto LeakyRelu =
            std::make_shared<ov::opset8::PRelu>(add2, ov::opset8::Constant::create(element::f32, Shape{1}, {1}));
        LeakyRelu->set_friendly_name("LeakyRelu");

        auto conv2 = create_conv(LeakyRelu);
        conv2->set_friendly_name("conv2");

        function = std::make_shared<ov::Model>(conv2, ParameterVector{input1, input2});
    }

    void TearDown() override {
        auto runtime_function = compiledModel.get_runtime_model();
        int nodes_found = 0;
        for (const auto& n : runtime_function->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "Subgraph") {
                nodes_found++;
                auto output_layout = n->get_rt_info().at(ov::exec_model_info::OUTPUT_LAYOUTS).as<std::string>();
                // The optimal choose should be: 'nhwc'.
                ASSERT_EQ(output_layout, "acdb");
            }
        }
        ASSERT_GT(nodes_found, 0);
    }
};

TEST_F(SubgraphSelectPD, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
