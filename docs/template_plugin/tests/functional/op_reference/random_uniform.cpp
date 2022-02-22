// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/opsets/opset8.hpp"

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
        const auto min_val_param = std::make_shared<opset8::Parameter>(min_val.type, min_val.shape);
        const auto max_val_param = std::make_shared<opset8::Parameter>(max_val.type, max_val.shape);
        auto out_shape_ = std::make_shared<opset8::Constant>(element::i64, Shape{out_shape.size()}, out_shape);

        return std::make_shared<ov::Model>(NodeVector{std::make_shared<opset8::RandomUniform>(out_shape_,
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
                            reference_tests::Tensor{{3, 2, 4},
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
                            reference_tests::Tensor{{3, 2, 4},
                                   element::f16,
                                   std::vector<float16>{0.60449219, 0.80664062, 0.83203125, 0.38378906, 0.03613281,
                                                        0.08300781, 0.54394531, 0.83398438, 0.33593750, 0.71972656,
                                                        0.15429688, 0.12890625, 0.34765625, 0.86914062, 0.41308594,
                                                        0.57226562, 0.57421875, 0.93945312, 0.65527344, 0.82226562,
                                                        0.82421875, 0.13281250, 0.64355469, 0.66015625}},
                            "float16_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{-650}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{450}},
                            element::Type_t::f32,
                            150,
                            10,
                            reference_tests::Tensor{{3, 2, 4},
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
                            reference_tests::Tensor{{3, 2, 4},
                                   element::f16,
                                   std::vector<float16>{-1.19726562, -1.09667969, -1.08398438, -1.30859375, -1.48242188,
                                                        -1.45898438, -1.22851562, -1.08300781, -1.33203125, -1.14062500,
                                                        -1.42285156, -1.43554688, -1.32617188, -1.06542969, -1.29296875,
                                                        -1.21386719, -1.21289062, -1.03027344, -1.17187500, -1.08886719,
                                                        -1.08789062, -1.43359375, -1.17773438, -1.16992188}},
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
                            reference_tests::Tensor{{5, 4, 3},
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
                            reference_tests::Tensor{{7, 3},
                                   element::bf16,
                                   std::vector<bfloat16>{0.8984375, 0.84375,   0.1640625, 0.1875,   0.46875,  0.6875,
                                                         0.5234375, 0.3046875, 0.9140625, 0.453125, 0.953125, 0.328125,
                                                         0.359375,  0.1875,    0.9453125, 0.390625, 0.21875,  0.9921875,
                                                         0.8203125, 0.453125,  0.875}},
                            "bfloat16_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{7, 3},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{-150}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{200}},
                            element::Type_t::bf16,
                            4978,
                            5164,
                            reference_tests::Tensor{{7, 3},
                                   element::bf16,
                                   std::vector<bfloat16>{164, 146, -92.5, -84.5, 14,  90,    33,  -43.5, 170, 8,  182,
                                                         -35, -24, -84.5, 180,   -14, -73.5, 198, 138,   8,   156}},
                            "bfloat16_non_default_min_max")),
    ReferenceRandomUniformLayerTest::getTestCaseName);
}  // namespace reference_tests
