// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "base_reference_test.hpp"
#include "openvino/op/eye.hpp"

using namespace ov;
using namespace reference_tests;

namespace reference_tests {
namespace {

struct EyeDefaultParams {
    EyeDefaultParams(const reference_tests::Tensor& num_rows,
                     const element::Type& output_type,
                     const reference_tests::Tensor& expected_tensor,
                     const std::string& test_case_name)
                     : num_rows(num_rows),
                     output_type(output_type),
                     expected_tensor(expected_tensor),
                     test_case_name(test_case_name) {}

    reference_tests::Tensor num_rows;
    element::Type output_type;
    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
};

struct EyeParams {
    EyeParams(const reference_tests::Tensor& num_rows,
              const reference_tests::Tensor& num_columns,
              const reference_tests::Tensor& diagonal_index,
              const element::Type& output_type,
              const reference_tests::Tensor& expected_tensor,
              const std::string& test_case_name)
              : num_rows(num_rows),
              num_columns(num_columns),
              diagonal_index(diagonal_index),
              output_type(output_type),
              expected_tensor(expected_tensor),
              test_case_name(test_case_name) {}

    reference_tests::Tensor num_rows;
    reference_tests::Tensor num_columns;
    reference_tests::Tensor diagonal_index;
    element::Type output_type;
    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
};

struct EyeBatchShapeParams {
    EyeBatchShapeParams(const reference_tests::Tensor& num_rows,
                        const reference_tests::Tensor& num_columns,
                        const reference_tests::Tensor& diagonal_index,
                        const reference_tests::Tensor& batch_shape,
                        const element::Type& output_type,
                        const reference_tests::Tensor& expected_tensor,
                        const std::string& test_case_name)
                        : num_rows(num_rows),
                        num_columns(num_columns),
                        diagonal_index(diagonal_index),
                        batch_shape(batch_shape),
                        output_type(output_type),
                        expected_tensor(expected_tensor),
                        test_case_name(test_case_name) {}

    reference_tests::Tensor num_rows;
    reference_tests::Tensor num_columns;
    reference_tests::Tensor diagonal_index;
    reference_tests::Tensor batch_shape;
    element::Type output_type;
    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
};

class ReferenceEyeDefaultLayerTest : public testing::TestWithParam<EyeDefaultParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.num_rows, params.output_type);
        inputData = {params.num_rows.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EyeDefaultParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& num_rows,
                                                 const element::Type& output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(num_rows.type, num_rows.shape);
        const auto Eye = std::make_shared<op::v9::Eye>(in, output_type);
        return std::make_shared<Model>(NodeVector{Eye}, ParameterVector{in});
    }
};

class ReferenceEyeLayerTest : public testing::TestWithParam<EyeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.num_rows,
                                  params.num_columns,
                                  params.diagonal_index,
                                  params.output_type);
        inputData = {params.num_rows.data, params.num_columns.data, params.diagonal_index.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EyeParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& num_rows,
                                                 const reference_tests::Tensor& num_columns,
                                                 const reference_tests::Tensor& diagonal_index,
                                                 const element::Type& output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(num_rows.type, num_rows.shape);
        const auto in2 = std::make_shared<op::v0::Parameter>(num_columns.type, num_columns.shape);
        const auto in3 = std::make_shared<op::v0::Parameter>(diagonal_index.type, diagonal_index.shape);
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
                                  params.output_type);
        inputData = {params.num_rows.data, params.num_columns.data, params.diagonal_index.data, params.batch_shape.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EyeBatchShapeParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& num_rows,
                                                 const reference_tests::Tensor& num_columns,
                                                 const reference_tests::Tensor& diagonal_index,
                                                 const reference_tests::Tensor& batch_shape,
                                                 const element::Type& output_type) {
        const auto in1 = std::make_shared<op::v0::Parameter>(num_rows.type, num_rows.shape);
        const auto in2 = std::make_shared<op::v0::Parameter>(num_columns.type, num_columns.shape);
        const auto in3 = std::make_shared<op::v0::Parameter>(diagonal_index.type, diagonal_index.shape);
        const auto in4 = std::make_shared<op::v0::Parameter>(batch_shape.type, batch_shape.shape);
        const auto Eye = std::make_shared<op::v9::Eye>(in1, in2, in3, in4, output_type);
        return std::make_shared<Model>(NodeVector{Eye}, ParameterVector{in1, in2, in3, in4});
    }
};


TEST_P(ReferenceEyeDefaultLayerTest, EyeDefaultWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceEyeLayerTest, EyeWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceEyeBatchShapeLayerTest, EyeRectangleBatchShapeWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
        smoke_EyeDefault_With_Hardcoded_Refs,
        ReferenceEyeDefaultLayerTest,
        ::testing::Values(
                EyeDefaultParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                                 element::Type_t::f32,
                                 reference_tests::Tensor{{3, 3}, element::f32, std::vector<float>{1, 0, 0,
                                                                                                  0, 1, 0,
                                                                                                  0, 0, 1}},
                                 "float32_3x3"),
                EyeDefaultParams(reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{2}},
                                 element::Type_t::u8,
                                 reference_tests::Tensor{{2, 2}, element::u8, std::vector<uint8_t>{1, 0,
                                                                                                   0, 1}},
                                 "int32_empty_2x2")),
        ReferenceEyeDefaultLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Eye_With_Hardcoded_Refs,
        ReferenceEyeLayerTest,
        ::testing::Values(
                EyeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                          reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                          reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
                          element::Type_t::f32,
                          reference_tests::Tensor{{3, 2}, element::f32, std::vector<float>{1, 0,
                                                                                           0, 1,
                                                                                           0, 0}},
                          "float32_default_3x2"),
                EyeParams(reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{2}},
                          reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{4}},
                          reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{2}},
                          element::Type_t::i8,
                          reference_tests::Tensor{{2, 4}, element::i8, std::vector<int8_t>{0, 0, 1, 0,
                                                                                           0, 0, 0, 1}},
                          "int8_diag=2_2x4"),
                EyeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
                          reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                          reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{-3}},
                          element::Type_t::i64,
                          reference_tests::Tensor{{4, 2}, element::i64, std::vector<int64_t>{0, 0,
                                                                                             0, 0,
                                                                                             0, 0,
                                                                                             1, 0}},
                          "int64_diag=-3_4x2"),
                EyeParams(reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{1}},
                          reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{4}},
                          reference_tests::Tensor{{}, element::i32, std::vector<int32_t>{10}},
                          element::Type_t::i8,
                          reference_tests::Tensor{{1, 4}, element::i8, std::vector<int8_t>{0, 0, 0, 0}},
                          "int8_empty_1x4")),
        ReferenceEyeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_EyeBatchShape_With_Hardcoded_Refs,
        ReferenceEyeBatchShapeLayerTest,
        ::testing::Values(
                EyeBatchShapeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{0}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                                    element::Type_t::f32,
                                    reference_tests::Tensor{{2, 3, 3}, element::f32, std::vector<float>{1, 0, 0,
                                                                                                        0, 1, 0,
                                                                                                        0, 0, 1,
                                                                                                        1, 0, 0,
                                                                                                        0, 1, 0,
                                                                                                        0, 0, 1}},
                                   "float32_2x3x3"),
                EyeBatchShapeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{4}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                                    reference_tests::Tensor{{2}, element::i32, std::vector<int32_t>{2, 2}},
                                    element::Type_t::i64,
                                    reference_tests::Tensor{{2, 3, 3}, element::i64, std::vector<int64_t>{0, 0, 1, 0,
                                                                                                          0, 0, 0, 1,
                                                                                                          0, 0, 1, 0,
                                                                                                          0, 0, 0, 1,
                                                                                                          0, 0, 1, 0,
                                                                                                          0, 0, 0, 1,
                                                                                                          0, 0, 1, 0,
                                                                                                          0, 0, 0, 1}},
                                    "int64_2x2x3x3"),
                EyeBatchShapeParams(reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{2}},
                                    reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{3}},
                                    reference_tests::Tensor{{2}, element::i32, std::vector<int32_t>{1, 3}},
                                    element::Type_t::u8,
                                    reference_tests::Tensor{{2, 3, 3}, element::u8, std::vector<uint8_t>{0, 0,
                                                                                                         0, 0,
                                                                                                         0, 0,
                                                                                                         0, 0,
                                                                                                         0, 0,
                                                                                                         0, 0}},
                                    "uint8_1x3x2x2")),
        ReferenceEyeBatchShapeLayerTest::getTestCaseName);

}  // namespace reference_tests 