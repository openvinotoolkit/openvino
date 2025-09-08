// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/loop.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {
enum LOOP_IN_TYPE { INVARIANT, MERGED };

struct LoopFunctionalBase {
    virtual std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& loop_inputs,
                                                       const std::vector<reference_tests::Tensor>& results,
                                                       const int64_t& trip_count_value = 1,
                                                       const std::vector<LOOP_IN_TYPE>& loop_in_type = {},
                                                       const ov::element::Type& net_type = ov::element::f32) = 0;
    LoopFunctionalBase() = default;
    virtual ~LoopFunctionalBase() = default;
};

struct LoopDynamicInputs : public LoopFunctionalBase {
    std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& loop_inputs,
                                               const std::vector<reference_tests::Tensor>& results,
                                               const int64_t& trip_count_value,
                                               const std::vector<LOOP_IN_TYPE>& loop_in_type,
                                               const ov::element::Type& net_type) override {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto M = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        auto Xi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto Yi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto M_body = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto body_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);

        auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, 3);
        auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        // Body
        auto sum = std::make_shared<ov::op::v1::Add>(Xi, Yi);
        auto Zo = std::make_shared<ov::op::v1::Multiply>(sum, M_body);
        auto body =
            std::make_shared<ov::Model>(ov::OutputVector{body_condition, Zo}, ov::ParameterVector{Xi, Yi, M_body});

        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_condition);
        loop->set_function(body);

        loop->set_invariant_input(Xi, X);
        loop->set_invariant_input(Yi, Y);
        loop->set_merged_input(M_body, M, Zo);

        loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

        // Output is last Zo
        auto result = std::make_shared<ov::op::v0::Result>(loop->get_iter_value(Zo, -1));
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
    ::testing::Values(LoopParams(
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

struct LoopStaticInputs : public LoopFunctionalBase {
    std::shared_ptr<ov::Model> create_function(const std::vector<reference_tests::Tensor>& loop_inputs,
                                               const std::vector<reference_tests::Tensor>& results,
                                               const int64_t& trip_count,
                                               const std::vector<LOOP_IN_TYPE>& loop_in_type,
                                               const ov::element::Type& net_type) override {
        ov::ParameterVector loop_params;
        for (auto&& input : loop_inputs) {
            loop_params.emplace_back(std::make_shared<ov::op::v0::Parameter>(input.type, input.shape));
        }

        // Set up the cell body, a function from (Xi, Yi) -> (Zo)
        // Body parameters
        const std::vector<ov::PartialShape> body_params_shapes(loop_inputs.size(), ov::PartialShape::dynamic());
        ov::ParameterVector body_params;
        for (const auto& pshape : body_params_shapes) {
            body_params.emplace_back(std::make_shared<ov::op::v0::Parameter>(net_type, pshape));
        }

        const auto body_condition_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        const auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        std::shared_ptr<ov::Node> trip_count_input;
        trip_count_input = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, trip_count);

        // Body
        std::shared_ptr<ov::Node> Zo = body_params[0];
        for (size_t i = 1; i < body_params.size(); ++i) {
            Zo = std::make_shared<ov::op::v1::Add>(body_params[i], Zo);
        }

        const auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition_const, Zo}, body_params);

        const auto loop = std::make_shared<ov::op::v5::Loop>(trip_count_input, exec_condition);
        loop->set_function(body);
        loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

        for (size_t i = 0; i < body_params.size(); ++i) {
            if (loop_in_type[i] == LOOP_IN_TYPE::INVARIANT) {
                loop->set_invariant_input(body_params[i], loop_params[i]);
            } else if (loop_in_type[i] == LOOP_IN_TYPE::MERGED) {
                // todo: support several merged loop_inputs
                // now supported only one in this sample
                loop->set_merged_input(body_params[i], loop_params[i], Zo);
            }
        }

        // Output 0 is last Zo
        const auto out0 = loop->get_iter_value(body_condition_const, -1);
        const auto out1 = loop->get_iter_value(Zo, -1);
        // Output 1 is concat of Zos
        // start=0, stride=1, part_size=1, end=-1, axis=1
        const auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

        const auto result0 = std::make_shared<ov::op::v0::Result>(out0);
        const auto result1 = std::make_shared<ov::op::v0::Result>(out1);
        const auto result2 = std::make_shared<ov::op::v0::Result>(out2);
        const auto function =
            std::make_shared<ov::Model>(ov::ResultVector{result0, result1, result2}, loop_params, "loop");
        return function;
    }
};

struct LoopStaticParams {
    LoopStaticParams(const std::shared_ptr<LoopFunctionalBase>& functional,
                     const std::vector<reference_tests::Tensor>& loop_inputs,
                     const std::vector<reference_tests::Tensor>& expected_results,
                     const int64_t& trip_count,
                     const std::vector<LOOP_IN_TYPE>& loop_in_type,
                     const ov::element::Type& net_type,
                     const std::string& test_case_name)
        : function(functional),
          inputs(loop_inputs),
          expected_results(expected_results),
          trip_count(trip_count),
          loop_in_type(loop_in_type),
          net_type(net_type),
          test_case_name(test_case_name) {}

    std::shared_ptr<LoopFunctionalBase> function;
    std::vector<reference_tests::Tensor> inputs;
    std::vector<reference_tests::Tensor> expected_results;
    int64_t trip_count;
    std::vector<LOOP_IN_TYPE> loop_in_type;
    ov::element::Type net_type;
    std::string test_case_name;
};

class ReferenceLoopLayerStaticTest : public testing::TestWithParam<LoopStaticParams>,
                                     public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        auto params = GetParam();
        function = params.function->create_function(params.inputs,
                                                    params.expected_results,
                                                    params.trip_count,
                                                    params.loop_in_type,
                                                    params.net_type);
        inputData.reserve(params.inputs.size());
        refOutData.reserve(params.expected_results.size());
        for (auto& input : params.inputs) {
            inputData.push_back(input.data);
        }
        for (auto& output : params.expected_results) {
            refOutData.push_back(output.data);
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<LoopStaticParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "TS=";
        for (auto& input : param.inputs) {
            result << ov::test::utils::vec2str(input.shape) << "_";
        }
        result << "_tripCount=" << param.trip_count;
        result << "_loopInType=";
        for (auto& type : param.loop_in_type) {
            result << "_" << type;
        }
        result << "_netType=" << param.net_type;
        if (!param.test_case_name.empty()) {
            result << "_" << param.test_case_name;
        }
        return result.str();
    }
};

TEST_P(ReferenceLoopLayerStaticTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t ET>
std::vector<LoopStaticParams> generateParams() {
    using T = typename ov::element_type_traits<ET>::value_type;
    std::vector<LoopStaticParams> params{
        LoopStaticParams(
            std::make_shared<LoopStaticInputs>(),
            {reference_tests::Tensor(ET, {10, 1, 10}, std::vector<T>{7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7,
                                                                     4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4,
                                                                     5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5,
                                                                     2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2,
                                                                     3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3,
                                                                     0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2}),
             reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{7}),
             reference_tests::Tensor(ET, {10, 1, 10}, std::vector<T>{7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7,
                                                                     4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4,
                                                                     5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5,
                                                                     2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2,
                                                                     3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3,
                                                                     0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2})},
            {reference_tests::Tensor(ov::element::Type_t::boolean, {1}, std::vector<char>{1}),
             reference_tests::Tensor(
                 ET,
                 {10, 1, 10},
                 std::vector<T>{21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19,
                                21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19,
                                21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11}),
             reference_tests::Tensor(
                 ET,
                 {10, 1, 10},
                 std::vector<T>{21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19,
                                21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19,
                                21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11})},
            1,
            {LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::MERGED},
            ET,
            "loop_for_common"),

        LoopStaticParams(
            std::make_shared<LoopStaticInputs>(),
            {reference_tests::Tensor(ET, {10, 1, 10}, std::vector<T>{7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7,
                                                                     4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4,
                                                                     5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5,
                                                                     2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2,
                                                                     3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3,
                                                                     0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2}),
             reference_tests::Tensor(ET, {1, 1, 1}, std::vector<T>{7}),
             reference_tests::Tensor(ET, {10, 1, 10}, std::vector<T>{7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7,
                                                                     4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4,
                                                                     5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5,
                                                                     2, 3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2,
                                                                     3, 0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2, 3,
                                                                     0, 1, 6, 7, 4, 5, 2, 3, 0, 1, 6, 7, 4, 5, 2})},
            {reference_tests::Tensor(ov::element::Type_t::boolean, {1}, std::vector<char>{1}),
             reference_tests::Tensor(
                 ET,
                 {10, 1, 10},
                 std::vector<T>{77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47,
                                53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71,
                                77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47,
                                53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71,
                                77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47}),
             reference_tests::Tensor(
                 ET,
                 {10, 5, 10},
                 std::vector<T>{21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 35, 26, 29, 20, 23, 14, 17, 32, 35, 26,
                                49, 37, 41, 29, 33, 21, 25, 45, 49, 37, 63, 48, 53, 38, 43, 28, 33, 58, 63, 48,
                                77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                29, 20, 23, 14, 17, 32, 35, 26, 29, 20, 41, 29, 33, 21, 25, 45, 49, 37, 41, 29,
                                53, 38, 43, 28, 33, 58, 63, 48, 53, 38, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47,

                                13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  23, 14, 17, 32, 35, 26, 29, 20, 23, 14,
                                33, 21, 25, 45, 49, 37, 41, 29, 33, 21, 43, 28, 33, 58, 63, 48, 53, 38, 43, 28,
                                53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 9,  19, 21, 15, 17, 11, 13, 7,  9,  19,
                                17, 32, 35, 26, 29, 20, 23, 14, 17, 32, 25, 45, 49, 37, 41, 29, 33, 21, 25, 45,
                                33, 58, 63, 48, 53, 38, 43, 28, 33, 58, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71,

                                21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 35, 26, 29, 20, 23, 14, 17, 32, 35, 26,
                                49, 37, 41, 29, 33, 21, 25, 45, 49, 37, 63, 48, 53, 38, 43, 28, 33, 58, 63, 48,
                                77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                29, 20, 23, 14, 17, 32, 35, 26, 29, 20, 41, 29, 33, 21, 25, 45, 49, 37, 41, 29,
                                53, 38, 43, 28, 33, 58, 63, 48, 53, 38, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47,

                                13, 7,  9,  19, 21, 15, 17, 11, 13, 7,  23, 14, 17, 32, 35, 26, 29, 20, 23, 14,
                                33, 21, 25, 45, 49, 37, 41, 29, 33, 21, 43, 28, 33, 58, 63, 48, 53, 38, 43, 28,
                                53, 35, 41, 71, 77, 59, 65, 47, 53, 35, 9,  19, 21, 15, 17, 11, 13, 7,  9,  19,
                                17, 32, 35, 26, 29, 20, 23, 14, 17, 32, 25, 45, 49, 37, 41, 29, 33, 21, 25, 45,
                                33, 58, 63, 48, 53, 38, 43, 28, 33, 58, 41, 71, 77, 59, 65, 47, 53, 35, 41, 71,

                                21, 15, 17, 11, 13, 7,  9,  19, 21, 15, 35, 26, 29, 20, 23, 14, 17, 32, 35, 26,
                                49, 37, 41, 29, 33, 21, 25, 45, 49, 37, 63, 48, 53, 38, 43, 28, 33, 58, 63, 48,
                                77, 59, 65, 47, 53, 35, 41, 71, 77, 59, 17, 11, 13, 7,  9,  19, 21, 15, 17, 11,
                                29, 20, 23, 14, 17, 32, 35, 26, 29, 20, 41, 29, 33, 21, 25, 45, 49, 37, 41, 29,
                                53, 38, 43, 28, 33, 58, 63, 48, 53, 38, 65, 47, 53, 35, 41, 71, 77, 59, 65, 47})},
            5,
            {LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::INVARIANT, LOOP_IN_TYPE::MERGED},
            ET,
            "loop_for_common"),
    };
    return params;
}

std::vector<LoopStaticParams> generateCombinedParams() {
    const std::vector<std::vector<LoopStaticParams>> generatedParams{
        generateParams<ov::element::Type_t::i8>(),
        generateParams<ov::element::Type_t::i16>(),
        generateParams<ov::element::Type_t::i32>(),
        generateParams<ov::element::Type_t::i64>(),
        generateParams<ov::element::Type_t::u8>(),
        generateParams<ov::element::Type_t::u16>(),
        generateParams<ov::element::Type_t::u32>(),
        generateParams<ov::element::Type_t::u64>(),
        generateParams<ov::element::Type_t::bf16>(),
        generateParams<ov::element::Type_t::f16>(),
        generateParams<ov::element::Type_t::f32>(),
    };
    std::vector<LoopStaticParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Loop_With_Hardcoded_Refs,
                         ReferenceLoopLayerStaticTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceLoopLayerStaticTest::getTestCaseName);
}  // namespace
