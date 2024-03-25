// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace reference_tests {
namespace {

struct EyeParams {
    EyeParams(const reference_tests::Tensor& num_rows,
              const reference_tests::Tensor& num_columns,
              const reference_tests::Tensor& diagonal_index,
              const element::Type& output_type,
              const reference_tests::Tensor& expected_tensor,
              const std::string& test_case_name,
              bool is_dyn_shape_test = false)
        : num_rows(num_rows),
          num_columns(num_columns),
          diagonal_index(diagonal_index),
          output_type(output_type),
          expected_tensor(expected_tensor),
          test_case_name(test_case_name),
          set_dynamic_shape(is_dyn_shape_test) {}

    reference_tests::Tensor num_rows;
    reference_tests::Tensor num_columns;
    reference_tests::Tensor diagonal_index;
    element::Type output_type;
    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
    bool set_dynamic_shape = false;
};

struct EyeBatchShapeParams {
    EyeBatchShapeParams(const reference_tests::Tensor& num_rows,
                        const reference_tests::Tensor& num_columns,
                        const reference_tests::Tensor& diagonal_index,
                        const reference_tests::Tensor& batch_shape,
                        const element::Type& output_type,
                        const reference_tests::Tensor& expected_tensor,
                        const std::string& test_case_name,
                        bool is_dyn_shape_test = false)
        : num_rows(num_rows),
          num_columns(num_columns),
          diagonal_index(diagonal_index),
          batch_shape(batch_shape),
          output_type(output_type),
          expected_tensor(expected_tensor),
          test_case_name(test_case_name),
          set_dynamic_shape(is_dyn_shape_test) {}

    reference_tests::Tensor num_rows;
    reference_tests::Tensor num_columns;
    reference_tests::Tensor diagonal_index;
    reference_tests::Tensor batch_shape;
    element::Type output_type;
    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
    bool set_dynamic_shape = false;
};

class ReferenceEyeLayerTest : public testing::TestWithParam<EyeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.num_rows,
                                  params.num_columns,
                                  params.diagonal_index,
                                  params.output_type,
                                  params.set_dynamic_shape);
        inputData = {params.num_rows.data, params.num_columns.data, params.diagonal_index.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EyeParams>& obj) {
        return obj.param.test_case_name + (obj.param.set_dynamic_shape ? "_dyn_shape_inputs" : "");
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& num_rows,
                                                 const reference_tests::Tensor& num_columns,
                                                 const reference_tests::Tensor& diagonal_index,
                                                 const element::Type& output_type,
                                                 bool set_dynamic_shape = false) {
        const auto in1 =
            std::make_shared<op::v0::Parameter>(num_rows.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : num_rows.shape);
        const auto in2 =
            std::make_shared<op::v0::Parameter>(num_columns.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : num_columns.shape);
        const auto in3 =
            std::make_shared<op::v0::Parameter>(diagonal_index.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : diagonal_index.shape);
        const auto Eye = std::make_shared<op::v9::Eye>(in1, in2, in3, output_type);
        return std::make_shared<Model>(NodeVector{Eye}, ParameterVector{in1, in2, in3});
    }
};

class ReferenceEyeBatchShapeLayerTest : public testing::TestWithParam<EyeBatchShapeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.num_rows,
                                  params.num_columns,
                                  params.diagonal_index,
                                  params.batch_shape,
                                  params.output_type,
                                  params.set_dynamic_shape);
        inputData = {params.num_rows.data,
                     params.num_columns.data,
                     params.diagonal_index.data,
                     params.batch_shape.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EyeBatchShapeParams>& obj) {
        return obj.param.test_case_name + (obj.param.set_dynamic_shape ? "_dyn_shape_inputs" : "");
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& num_rows,
                                                 const reference_tests::Tensor& num_columns,
                                                 const reference_tests::Tensor& diagonal_index,
                                                 const reference_tests::Tensor& batch_shape,
                                                 const element::Type& output_type,
                                                 bool set_dynamic_shape = false) {
        const auto in1 =
            std::make_shared<op::v0::Parameter>(num_rows.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : num_rows.shape);
        const auto in2 =
            std::make_shared<op::v0::Parameter>(num_columns.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : num_columns.shape);
        const auto in3 =
            std::make_shared<op::v0::Parameter>(diagonal_index.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : diagonal_index.shape);
        const auto in4 =
            std::make_shared<op::v0::Parameter>(batch_shape.type,
                                                set_dynamic_shape ? PartialShape::dynamic() : batch_shape.shape);
        const auto Eye = std::make_shared<op::v9::Eye>(in1, in2, in3, in4, output_type);
        return std::make_shared<Model>(NodeVector{Eye}, ParameterVector{in1, in2, in3, in4});
    }
};

std::vector<EyeParams> generateEyeParams(bool is_dyn_shape_test = false) {
    std::vector<EyeParams> test_params{
        EyeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                  reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                  reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
                  element::Type_t::f32,
                  reference_tests::Tensor{{3, 2}, element::f32, std::vector<float>{1, 0, 0, 1, 0, 0}},
                  "float32_default_3x2",
                  is_dyn_shape_test),
        EyeParams(reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{2}},
                  reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{4}},
                  reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{2}},
                  element::Type_t::i8,
                  reference_tests::Tensor{{2, 4}, element::i8, std::vector<int8_t>{0, 0, 1, 0, 0, 0, 0, 1}},
                  "int8_diag=2_2x4",
                  is_dyn_shape_test),
        EyeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
                  reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                  reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{-3}},
                  element::Type_t::i64,
                  reference_tests::Tensor{{4, 2}, element::i64, std::vector<int64_t>{0, 0, 0, 0, 0, 0, 1, 0}},
                  "int64_diag=-3_4x2",
                  is_dyn_shape_test),
        EyeParams(reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{1}},
                  reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{4}},
                  reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{10}},
                  element::Type_t::i8,
                  reference_tests::Tensor{{1, 4}, element::i8, std::vector<int8_t>{0, 0, 0, 0}},
                  "int8_empty_1x4",
                  is_dyn_shape_test)};
    return test_params;
}

std::vector<EyeBatchShapeParams> generateEyeBatchShapeParams(bool is_dyn_shape_test = false) {
    std::vector<EyeBatchShapeParams> test_params{
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::f32,
            reference_tests::Tensor{{2, 3, 3},
                                    element::f32,
                                    std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1}},
            "f32_2x3x3_diag0",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::f32,
            reference_tests::Tensor{{2, 4, 4}, element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                                                                0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1,
                                                                                0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
            "f32_2x4x4_diag0",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::f32,
            reference_tests::Tensor{{2, 3, 4}, element::f32, std::vector<float>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
                                                                                1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
            "f32_2x3x4_diag0",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 4, 3}, element::i8, std::vector<int8_t>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                                                                1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}},
            "f32_2x4x3_diag0",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{1}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::f32,
            reference_tests::Tensor{{2, 3, 4}, element::f32, std::vector<float>{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                                                                                0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
            "f32_2x3x4_diag1",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{1}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::f32,
            reference_tests::Tensor{{2, 4, 3}, element::f32, std::vector<float>{0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                                                                                0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}},
            "f32_2x4x3_diag1",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 3, 4}, element::i8, std::vector<int8_t>{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                                                                0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
            "i8_2x3x4_diag2",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 4, 3}, element::i8, std::vector<int8_t>{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
            "i8_2x4x3_diag2",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-1}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 3, 4}, element::i8, std::vector<int8_t>{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                                                                                0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
            "i8_2x3x4_diag-1",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-1}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 4, 3}, element::i8, std::vector<int8_t>{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                                                                0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1}},
            "i8_2x4x3_diag-1",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-2}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 3, 4}, element::i8, std::vector<int8_t>{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                                                                0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
            "i8_2x3x4_diag-2",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-2}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i8,
            reference_tests::Tensor{{2, 4, 3}, element::i8, std::vector<int8_t>{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                                                                                0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0}},
            "i8_2x4x3_diag-2",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{6}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{5}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{1}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            element::Type_t::i32,
            reference_tests::Tensor{{2, 6, 5},
                                    element::i32,
                                    std::vector<int32_t>{0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
                                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                                                         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
            "int32_2x6x5_diag1",
            is_dyn_shape_test),
        EyeBatchShapeParams(
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
            reference_tests::Tensor{{2}, element::i32, std::vector<int32_t>{2, 2}},
            element::Type_t::i64,
            reference_tests::Tensor{{2, 2, 2, 4}, element::i64, std::vector<int64_t>{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1,
                                                                                     0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                                                                                     0, 1, 0, 0, 1, 0, 0, 0, 0, 1}},
            "int64_2x2x2x4",
            is_dyn_shape_test),
        EyeBatchShapeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                            reference_tests::Tensor{{2}, element::i32, std::vector<int32_t>{1, 3}},
                            element::Type_t::u8,
                            reference_tests::Tensor{{1, 3, 2, 2},
                                                    element::u8,
                                                    std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                            "uint8_1x3x2x2",
                            is_dyn_shape_test)};
    return test_params;
}

std::vector<EyeParams> generateEyeCombinedParams() {
    std::vector<EyeParams> combined_params = generateEyeParams(false);
    std::vector<EyeParams> dyn_shape_params = generateEyeParams(true);
    combined_params.insert(combined_params.end(), dyn_shape_params.begin(), dyn_shape_params.end());
    return combined_params;
}

std::vector<EyeBatchShapeParams> generateEyeBatchShapeCombinedParams() {
    std::vector<EyeBatchShapeParams> combined_params = generateEyeBatchShapeParams(false);
    std::vector<EyeBatchShapeParams> dyn_shape_params = generateEyeBatchShapeParams(true);
    combined_params.insert(combined_params.end(), dyn_shape_params.begin(), dyn_shape_params.end());
    return combined_params;
}

TEST_P(ReferenceEyeLayerTest, EyeWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceEyeBatchShapeLayerTest, EyeRectangleBatchShapeWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_Eye_With_Hardcoded_Refs,
                         ReferenceEyeLayerTest,
                         // Generate params (3 inputs) with static and dynamic shapes
                         ::testing::ValuesIn(generateEyeCombinedParams()),
                         ReferenceEyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EyeBatchShape_With_Hardcoded_Refs,
                         ReferenceEyeBatchShapeLayerTest,
                         // Generate params (4 inputs) with static and dynamic shapes
                         ::testing::ValuesIn(generateEyeBatchShapeCombinedParams()),
                         ReferenceEyeBatchShapeLayerTest::getTestCaseName);

}  // namespace reference_tests
