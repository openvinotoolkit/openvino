// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
namespace ov {
namespace test {

/*  Bf16 saturation handling for the PowerStatic gamma parameter when input precision is bf16

       Param0     Param1
         |          |
       matmul0     matmul1
         \          /
          \        /
           concat
              |
              |  /const(-3.40282e+38)
            add /
              |
              |
            result

    This pattern creates a PowerStatic node with gamma parameter = -3.40282e+38,
    which exceeds bf16 range and requires saturation handling.
*/

class PowerStaticBF16Saturation : virtual public SubgraphBaseTest, public CpuTestWithFusing {
protected:
    void SetUp() override {
        abs_threshold = 0;
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape input0 = {{-1, -1}, {{1, 1}}};
        InputShape input1 = {{-1, -1}, {{1, 1}}};

        init_input_shapes({input0, input1});
        ov::ParameterVector parameters;
        auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
        parameters.push_back(param0);
        parameters.push_back(param1);

        // Constants that will create large values exceeding bf16 range
        const auto const_large_positive = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                                 ov::Shape{1, 1},
                                                                                 std::vector<float>({3.40282e+38f}));
        const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(parameters[0], const_large_positive, false, false);
        const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(parameters[1], const_large_positive, false, false);
        const auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{matMul0, matMul1}, 0);

        const auto const_large_negative =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, -3.40282e+38f);
        const auto add = std::make_shared<ov::op::v1::Add>(concat, const_large_negative);

        function = makeNgraphFunction(ElementType::f32, parameters, add, "PowerStaticBF16Saturation");
        configuration.insert({ov::hint::inference_precision(ov::element::bf16)});
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 1;
        in_data.range = 1;
        in_data.resolution = 1;

        for (size_t i = 0; i < modelInputs.size(); ++i) {
            auto tensor = ov::test::utils::create_and_fill_tensor(modelInputs[i].get_element_type(),
                                                                  targetInputStaticShapes[i],
                                                                  in_data);
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(PowerStaticBF16Saturation, CompareWithRefs) {
    run();
}

/*  Bf16 saturation handling for the PowerStatic beta parameter when input precision is bf16

       Param0     Param1
         |          |
       matmul0     matmul1
         \          /
          \        /
           concat
              |
           multiply(1.0)
              |
           subtract(1.0 - multiply_result)
              |
           multiply(-3.40282e+38)
              |
            result

    This pattern gets fused into a PowerStatic node with beta parameter = -3.40282e+38,
    which exceeds bf16 range and requires saturation handling.
    The matmul operations ensure the input precision is bf16 through inference_precision setting.
*/

class PowerStaticBetaBF16Saturation : virtual public SubgraphBaseTest, public CpuTestWithFusing {
protected:
    void SetUp() override {
        abs_threshold = 0;
        targetDevice = ov::test::utils::DEVICE_CPU;
        InputShape input0 = {{-1, -1}, {{1, 1}}};
        InputShape input1 = {{-1, -1}, {{1, 1}}};

        init_input_shapes({input0, input1});
        ov::ParameterVector parameters;
        auto param0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
        parameters.push_back(param0);
        parameters.push_back(param1);

        // Constants that create a PowerStatic node with beta parameter exceeding bf16 range
        const auto const_one =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{1.0f});

        const auto const_large_negative =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                   ov::Shape{1, 1},
                                                   std::vector<float>{-3.40282346638528859812e+38f});

        const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(parameters[0], const_one, false, false);
        const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(parameters[1], const_one, false, false);
        const auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{matMul0, matMul1}, 0);

        // Create computation pattern:
        // Step 1: concat_result * 1.0 = concat_result (bf16 after precision conversion)
        // Step 2: 1.0 - multiply_result
        // Step 3: (1.0 - multiply_result) * (-3.40282e+38)
        // This pattern will be fused into PowerStatic node with beta = -3.40282e+38, exceeding bf16 range
        auto multiply_step = std::make_shared<ov::op::v1::Multiply>(concat, const_one);
        auto subtract_step = std::make_shared<ov::op::v1::Subtract>(const_one, multiply_step);
        auto final_multiply = std::make_shared<ov::op::v1::Multiply>(subtract_step, const_large_negative);

        function = makeNgraphFunction(ElementType::f32, parameters, final_multiply, "PowerStaticBetaBF16Saturation");
        configuration.insert({ov::hint::inference_precision(ov::element::bf16)});
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& modelInputs = function->inputs();
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 1;
        in_data.range = 1;
        in_data.resolution = 1;

        for (size_t i = 0; i < modelInputs.size(); ++i) {
            auto tensor = ov::test::utils::create_and_fill_tensor(modelInputs[i].get_element_type(),
                                                                  targetInputStaticShapes[i],
                                                                  in_data);
            inputs.insert({modelInputs[i].get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(PowerStaticBetaBF16Saturation, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
