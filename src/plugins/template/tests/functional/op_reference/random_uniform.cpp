// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_uniform.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

namespace reference_tests {
namespace {

struct RandomUniformParams {
    RandomUniformParams(const std::vector<int64_t>& paramOutShape,
                        const reference_tests::Tensor& paramMinValue,
                        const reference_tests::Tensor& paramMaxValue,
                        ov::element::Type paramOutType,
                        int64_t paramGlobalSeed,
                        int64_t paramOpSeed,
                        const reference_tests::Tensor& paramExpected,
                        const std::string& test_name)
        : out_shape(paramOutShape),
          min_val(paramMinValue),
          max_val(paramMaxValue),
          out_type(paramOutType),
          global_seed(paramGlobalSeed),
          op_seed(paramOpSeed),
          expected(paramExpected),
          test_case_name(test_name) {}
    std::vector<int64_t> out_shape;
    reference_tests::Tensor min_val;
    reference_tests::Tensor max_val;
    ov::element::Type out_type;
    int64_t global_seed;
    int64_t op_seed;
    reference_tests::Tensor expected;
    std::string test_case_name;
};

class ReferenceRandomUniformLayerTest : public testing::TestWithParam<RandomUniformParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.out_shape,
                                  params.min_val,
                                  params.max_val,
                                  params.out_type,
                                  params.global_seed,
                                  params.op_seed);
        inputData = {params.min_val.data, params.max_val.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const std::vector<int64_t>& out_shape,
                                                 const reference_tests::Tensor& min_val,
                                                 const reference_tests::Tensor& max_val,
                                                 const ov::element::Type& out_type,
                                                 int64_t global_seed,
                                                 int64_t op_seed) {
        const auto min_val_param = std::make_shared<op::v0::Parameter>(min_val.type, min_val.shape);
        const auto max_val_param = std::make_shared<op::v0::Parameter>(max_val.type, max_val.shape);
        auto out_shape_ = std::make_shared<op::v0::Constant>(element::i64, Shape{out_shape.size()}, out_shape);

        return std::make_shared<ov::Model>(NodeVector{std::make_shared<op::v8::RandomUniform>(out_shape_,
                                                                                              min_val_param,
                                                                                              max_val_param,
                                                                                              out_type,
                                                                                              global_seed,
                                                                                              op_seed)},
                                           ParameterVector{min_val_param, max_val_param});
    }
};

TEST_P(ReferenceRandomUniformLayerTest, RandomUniformWithHardcodedRefs) {
    Exec();
}

}  // namespace

// Reference values for the following tests are obtained from single layer TensorFlow model with tf.random.uniform().
INSTANTIATE_TEST_SUITE_P(
    smoke_RandomUniform_With_Hardcoded_Refs,
    ReferenceRandomUniformLayerTest,
    ::testing::Values(
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{0}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{1}},
                            element::Type_t::f32,
                            150,
                            10,
                            reference_tests::Tensor{
                                {3, 2, 4},
                                element::f32,
                                std::vector<float>{0.70112360, 0.30539632, 0.93931055, 0.94560349, 0.11694777,
                                                   0.50770056, 0.51971972, 0.22727466, 0.99137402, 0.35519040,
                                                   0.82692313, 0.59864855, 0.31364107, 0.57481313, 0.41399086,
                                                   0.96308255, 0.37140799, 0.85253167, 0.09358585, 0.08200955,
                                                   0.23655081, 0.81056309, 0.74226606, 0.76106691}},
                            "float32_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{0}},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{1}},
                            element::Type_t::f16,
                            150,
                            10,
                            reference_tests::Tensor{
                                {3, 2, 4},
                                element::f16,
                                std::vector<float16>{0.70112360, 0.30539632, 0.93931055, 0.94560349, 0.11694777,
                                                     0.50770056, 0.51971972, 0.22727466, 0.99137402, 0.35519040,
                                                     0.82692313, 0.59864855, 0.31364107, 0.57481313, 0.41399086,
                                                     0.96308255, 0.37140799, 0.85253167, 0.09358585, 0.08200955,
                                                     0.23655081, 0.81056309, 0.74226606, 0.76106691}},
                            "float16_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{-650}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{450}},
                            element::Type_t::f32,
                            150,
                            10,
                            reference_tests::Tensor{
                                {3, 2, 4},
                                element::f32,
                                std::vector<float>{121.23596191,  -314.06405640, 383.24157715,  390.16381836,
                                                   -521.35742188, -91.52935791,  -78.30828857,  -399.99786377,
                                                   440.51147461,  -259.29055786, 259.61541748,  8.51342773,
                                                   -304.99481201, -17.70556641,  -194.61004639, 409.39074707,
                                                   -241.45120239, 287.78485107,  -547.05554199, -559.78948975,
                                                   -389.79409790, 241.61938477,  166.49267578,  187.17358398}},
                            "float32_non_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{-1.5}},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{-1.0}},
                            element::Type_t::f16,
                            150,
                            10,
                            reference_tests::Tensor{
                                {3, 2, 4},
                                element::f16,
                                std::vector<float16>{-1.14941, -1.34766, -1.03027, -1.02734, -1.44141, -1.24609,
                                                     -1.24023, -1.38672, -1.00391, -1.32227, -1.08691, -1.2002,
                                                     -1.34277, -1.21289, -1.29297, -1.01855, -1.31445, -1.07422,
                                                     -1.45312, -1.45898, -1.38184, -1.09473, -1.12891, -1.11914}},
                            "float16_non_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{2, 3, 4},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-100}},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{50}},
                            element::Type_t::i32,
                            100,
                            350,
                            reference_tests::Tensor{{2, 3, 4},
                                                    element::i32,
                                                    std::vector<int32_t>{
                                                        22, -56, -33, -89, -98, -33, -3,  -48, -82, 5,  -66, 21,
                                                        29, -42, -73, -37, 3,   36,  -35, 20,  -11, -8, -78, 47,
                                                    }},
                            "int32"),
        RandomUniformParams(std::vector<int64_t>{5, 4, 3},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{-2600}},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{3700}},
                            element::Type_t::i64,
                            755,
                            951,
                            reference_tests::Tensor{
                                {5, 4, 3},
                                element::i64,
                                std::vector<int64_t>{
                                    2116, -1581, 2559,  -339,  -1660, 519,   90,   2027,  -210, 3330, 1831,  -1737,
                                    2683, 2661,  3473,  1220,  3534,  -2384, 2199, 1935,  499,  2861, 2743,  3223,
                                    -531, -836,  -65,   3435,  632,   1765,  2613, 1891,  1698, 3069, 169,   -792,
                                    -32,  2976,  -1552, -2588, 3327,  -1756, 2637, -1084, 3567, -778, -1465, 2967,
                                    1242, 2672,  -1585, -2271, 3536,  -1502, 400,  2241,  3126, 908,  1073,  -2110}},
                            "int64"),
        RandomUniformParams(std::vector<int64_t>{7, 3},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{0}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{1}},
                            element::Type_t::bf16,
                            4978,
                            5164,
                            reference_tests::Tensor{
                                {7, 3},
                                element::bf16,
                                std::vector<bfloat16>{0.324219, 0.472656, 0.863281, 0.273438, 0.00518799, 0.0683594,
                                                      0.447266, 0.5625,   0.773438, 0.789062, 0.375,      0.0424805,
                                                      0.484375, 0.8125,   0.296875, 0.367188, 0.570312,   0.209961,
                                                      0.867188, 0.265625, 0.855469}},
                            "bfloat16_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{7, 3},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{-150}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{200}},
                            element::Type_t::bf16,
                            4978,
                            5164,
                            reference_tests::Tensor{
                                {7, 3},
                                element::bf16,
                                std::vector<bfloat16>{-36.5, 15.4375, 152, -54,    -148, -126,   6.59375,
                                                      47,    121.5,   126, -18.25, -135, 20,     134,
                                                      -45.5, -21.5,   49,  -76,    154,  -57.25, 150}},
                            "bfloat16_non_default_min_max")),
    ReferenceRandomUniformLayerTest::getTestCaseName);
}  // namespace reference_tests
