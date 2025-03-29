// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

/*This test runs the following subgraph:


    Param_0    Param_1
      \          |
       \       Softmax
        \        /
         \      /
          Concat
            |
          Softmax
            |
          Output_1

    Output_1 -> Param_1

  The main purpose of this test is checking the code path when the output tensor is reused as an input tensor of the
  next infer request.
*/

class InputOutputTensorReuse : public SubgraphBaseTest {
public:
    void SetUp() override {
        constexpr size_t softmax_axis = 1ul;
        constexpr int concat_axis = 2;
        targetDevice = ov::test::utils::DEVICE_CPU;
        auto netPrc = ov::element::f32;

        ov::ParameterVector input_params;
        input_params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrc, ov::PartialShape{1, 32, -1, 16}));
        input_params.push_back(std::make_shared<ov::op::v0::Parameter>(netPrc, ov::PartialShape{1, 32, -1, 16}));
        input_params[0]->set_friendly_name("Param_0");
        input_params[1]->set_friendly_name("Param_1");

        auto first_soft_max = std::make_shared<ov::op::v1::Softmax>(input_params[1], softmax_axis);
        auto concat =
            std::make_shared<ov::op::v0::Concat>(ov::NodeVector{input_params[0], first_soft_max}, concat_axis);
        auto last_soft_max = std::make_shared<ov::op::v1::Softmax>(concat, softmax_axis);

        ov::ResultVector results;
        for (size_t i = 0; i < last_soft_max->get_output_size(); i++)
            results.push_back(std::make_shared<ov::op::v0::Result>(last_soft_max->output(i)));

        results.front()->set_friendly_name("Output_1");

        function = std::make_shared<ov::Model>(results, input_params, "InputOutputTensorReuseTest");
    }
};

TEST_F(InputOutputTensorReuse, smoke_Input_Output_Binding) {
    compile_model();
    std::vector<ov::Shape> inputShapes = {{1, 32, 5, 16}, {1, 32, 1, 16}};
    generate_inputs(inputShapes);
    validate();

    constexpr size_t num_iter = 10;
    for (size_t i = 0; i < num_iter; i++) {
        auto outputTensor = inferRequest.get_output_tensor(0);
        inputShapes.back() = outputTensor.get_shape();
        auto itr = std::find_if(inputs.begin(),
                                inputs.end(),
                                [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& item) {
                                    return item.first->get_friendly_name() == "Param_1";
                                });
        ASSERT_NE(itr, inputs.end());
        itr->second = outputTensor;
        const auto& expectedOutputs = calculate_refs();

        for (const auto& input : inputs) {
            inferRequest.set_tensor(input.first, input.second);
        }
        inferRequest.infer();
        compare(expectedOutputs, {outputTensor});
    }
}

TEST_F(InputOutputTensorReuse, smoke_Input_Output_Bind_Once) {
    compile_model();
    std::vector<ov::Shape> inputShapes = {{1, 32, 5, 16}, {1, 32, 1, 16}};
    generate_inputs(inputShapes);
    validate();

    auto outputTensor = inferRequest.get_output_tensor(0);
    inputShapes.back() = outputTensor.get_shape();
    auto itr =
        std::find_if(inputs.begin(), inputs.end(), [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& item) {
            return item.first->get_friendly_name() == "Param_1";
        });
    ASSERT_NE(itr, inputs.end());
    itr->second = outputTensor;

    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }

    constexpr size_t num_iter = 10;
    for (size_t i = 0; i < num_iter; i++) {
        const auto& expectedOutputs = calculate_refs();

        inferRequest.infer();
        compare(expectedOutputs, {outputTensor});
        auto itr = std::find_if(inputs.begin(),
                                inputs.end(),
                                [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& item) {
                                    return item.first->get_friendly_name() == "Param_1";
                                });
        ASSERT_NE(itr, inputs.end());
        itr->second = expectedOutputs.front();
    }
}

TEST_F(InputOutputTensorReuse, smoke_Input_Output_Bind_Once_Empty_Tensor) {
    compile_model();
    std::vector<ov::Shape> inputShapes = {{1, 32, 5, 16}, {1, 32, 1, 16}};
    generate_inputs(inputShapes);
    inferRequest = compiledModel.create_infer_request();

    auto outputTensor = inferRequest.get_output_tensor(0);
    inputShapes.back() = outputTensor.get_shape();
    auto itr =
        std::find_if(inputs.begin(), inputs.end(), [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& item) {
            return item.first->get_friendly_name() == "Param_1";
        });
    ASSERT_NE(itr, inputs.end());
    itr->second = outputTensor;

    for (const auto& input : inputs) {
        inferRequest.set_tensor(input.first, input.second);
    }

    constexpr size_t num_iter = 10;
    for (size_t i = 0; i < num_iter; i++) {
        const auto& expectedOutputs = calculate_refs();

        inferRequest.infer();
        compare(expectedOutputs, {outputTensor});
        auto itr = std::find_if(inputs.begin(),
                                inputs.end(),
                                [](const std::pair<std::shared_ptr<ov::Node>, ov::Tensor>& item) {
                                    return item.first->get_friendly_name() == "Param_1";
                                });
        ASSERT_NE(itr, inputs.end());
        itr->second = expectedOutputs.front();
    }
}

}  // namespace test
}  // namespace ov