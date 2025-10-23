// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
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

        const auto constMatmul0 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                         ov::Shape{1, 1},
                                                                         std::vector<double>({3.40282e+38}));
        const auto constMatmul1 = std::make_shared<ov::op::v0::Constant>(ov::element::f32,
                                                                         ov::Shape{1, 1},
                                                                         std::vector<double>({3.40282e+38}));
        const auto matMul0 = std::make_shared<ov::op::v0::MatMul>(parameters[0], constMatmul0, false, false);
        const auto matMul1 = std::make_shared<ov::op::v0::MatMul>(parameters[1], constMatmul1, false, false);
        const auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{matMul0, matMul1}, 0);

        const auto constAdd = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, -3.40282e+38);
        const auto add = std::make_shared<ov::op::v1::Add>(concat, constAdd);

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
        auto tensor0 = ov::test::utils::create_and_fill_tensor(modelInputs[0].get_element_type(),
                                                               targetInputStaticShapes[0],
                                                               in_data);
        auto tensor1 = ov::test::utils::create_and_fill_tensor(modelInputs[1].get_element_type(),
                                                               targetInputStaticShapes[1],
                                                               in_data);

        inputs.insert({modelInputs[0].get_node_shared_ptr(), tensor0});
        inputs.insert({modelInputs[1].get_node_shared_ptr(), tensor1});
    }
};

TEST_F(PowerStaticBF16Saturation, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
