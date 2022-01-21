// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>

#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

struct TIFunctionalBase {
    virtual std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& ti_inputs,
                                                   const std::vector<reference_tests::Tensor>& results) = 0;
    TIFunctionalBase() = default;
    virtual ~TIFunctionalBase() = default;
};

struct TIDynamicInputs : public TIFunctionalBase {
    std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& ti_inputs,
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

        auto tensor_iterator = std::make_shared<ov::opset8::TensorIterator>();
        tensor_iterator->set_function(body);

        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_sliced_input(Yi, Y, 0, 1, 1, -1, 0);
        tensor_iterator->set_merged_input(M_body, M, Zo);

        // Output 0 is last Zo
        auto out1 = tensor_iterator->get_iter_value(Zo, -1);
        return std::make_shared<ov::Model>(ov::OutputVector{out1}, ov::ParameterVector{X, Y, M});
    }
};

struct TensorIteratorParams {
    TensorIteratorParams(const std::shared_ptr<TIFunctionalBase>& functional,
             const std::vector<reference_tests::Tensor>& ti_inputs,
             const std::vector<reference_tests::Tensor>& expected_results,
             const std::string& test_case_name)
            : function(functional),
              inputs(ti_inputs),
              expected_results(expected_results),
              test_case_name(test_case_name) {}

    std::shared_ptr<TIFunctionalBase> function;
    std::vector<reference_tests::Tensor> inputs;
    std::vector<reference_tests::Tensor> expected_results;
    std::string test_case_name;
};

class ReferenceTILayerTest : public testing::TestWithParam<TensorIteratorParams>,
        public reference_tests::CommonReferenceTest {
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
    static std::string getTestCaseName(const testing::TestParamInfo<TensorIteratorParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }
};

TEST_P(ReferenceTILayerTest, TensorIteratorWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
        smoke_TensorIterator_With_Hardcoded_Refs,
        ReferenceTILayerTest,
        ::testing::Values(
                TensorIteratorParams(
                        std::make_shared<TIDynamicInputs>(),
                        std::vector<reference_tests::Tensor>{
                            reference_tests::Tensor(ov::element::f32, ov::Shape{1, 2}, std::vector<float>{2, 3}),
                            reference_tests::Tensor(ov::element::f32, ov::Shape{2, 1}, std::vector<float>{4, 5}),
                            reference_tests::Tensor(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{5})},
                        std::vector<reference_tests::Tensor>{
                            reference_tests::Tensor(ov::element::f32, ov::Shape{1, 1}, std::vector<float>{240})},
                        "tensor_iterator_dynamic_inputs")),
        ReferenceTILayerTest::getTestCaseName);
