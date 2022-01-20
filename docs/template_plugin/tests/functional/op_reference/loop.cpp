// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

struct LoopFunctionalBase {
    virtual std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& loop_inputs,
                                                       const std::vector<reference_tests::Tensor>& results) = 0;
    LoopFunctionalBase() = default;
    virtual ~LoopFunctionalBase() = default;
};

struct LoopDynamicInputs : public LoopFunctionalBase {
    std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& loop_inputs,
                                               const std::vector<reference_tests::Tensor>& results) override {
        auto X = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto Y = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto M = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto Xi = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto Yi = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto M_body = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto body_condition = std::make_shared<ov::opset8::Constant>(ov::element::boolean, ov::Shape{1}, true);

        auto trip_count = std::make_shared<ov::opset8::Constant>(ngraph::element::i64, ov::Shape{1}, 3);
        auto exec_condition = std::make_shared<ov::opset8::Constant>(ngraph::element::boolean, ov::Shape{1}, true);
        // Body
        auto sum = std::make_shared<ov::opset8::Add>(Xi, Yi);
        auto Zo = std::make_shared<ov::opset8::Multiply>(sum, M_body);
        auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo},
                                                ov::ParameterVector{Xi, Yi, M_body});

        auto loop = std::make_shared<ov::opset8::Loop>(trip_count, exec_condition);
        loop->set_function(body);

        loop->set_invariant_input(Xi, X);
        loop->set_invariant_input(Yi, Y);
        loop->set_merged_input(M_body, M, Zo);

        loop->set_special_body_ports(ov::opset8::Loop::SpecialBodyPorts{-1, 0});

        // Output is last Zo
        auto result = std::make_shared<ov::opset8::Result>(loop->get_iter_value(Zo, -1));
        return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{X, Y, M});
    }
};

struct LoopParams {
    LoopParams(const std::shared_ptr<LoopFunctionalBase>& functional,
               const std::vector<reference_tests::Tensor>& loop_inputs,
               const std::vector<reference_tests::Tensor>& expected_results,
               const std::string& test_case_name)
            : function(functional),
              inputs(loop_inputs),
              expected_results(expected_results),
              test_case_name(test_case_name) {}

    std::shared_ptr<LoopFunctionalBase> function;
    std::vector<reference_tests::Tensor> inputs;
    std::vector<reference_tests::Tensor> expected_results;
    std::string test_case_name;
};

class ReferenceLoopLayerTest : public testing::TestWithParam<LoopParams>, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        auto params = GetParam();
        function = params.function->create_function(params.inputs, params.expected_results);
        inputData.reserve(params.inputs.size());
        refOutData.reserve(params.expected_results.size());
        for (auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        for (auto& expected_tensor : params.expected_results) {
            refOutData.push_back(expected_tensor.data);
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<LoopParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }
};

TEST_P(ReferenceLoopLayerTest, TensorIteratorWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
        smoke_TensorIterator_With_Hardcoded_Refs,
        ReferenceLoopLayerTest,
        ::testing::Values(
                LoopParams(
                        std::make_shared<LoopDynamicInputs>(),
                        std::vector<reference_tests::Tensor>{
                                reference_tests::Tensor(ov::element::f32, ov::Shape{2, 2}, std::vector<float>{0, 1, 2, 3}),
                                reference_tests::Tensor(ov::element::f32, ov::Shape{2, 2}, std::vector<float>{1, 2, 3, 4}),
                                reference_tests::Tensor(ov::element::f32, ov::Shape{2, 2}, std::vector<float>{5, 4, 3, 2})},
                        // 5*(0+1)*(0+1)*(0+1) = 5
                        // 4*(1+2)*(1+2)*(1+2) = 108
                        // 3*(2+3)*(2+3)*(2+3) = 375
                        // 2*(3+4)*(3+4)*(3+4) = 686
                        std::vector<reference_tests::Tensor>{
                                reference_tests::Tensor(ov::element::f32, ov::Shape{2, 2}, std::vector<float>{5, 108, 375, 686})},
                        "loop_dynamic_inputs")),
        ReferenceLoopLayerTest::getTestCaseName);
