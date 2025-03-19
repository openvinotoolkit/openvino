// Copyright (C) 2018-2025 Intel Corporation
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
    RandomUniformParams(const std::vector<int64_t>& out_shape,
                        const reference_tests::Tensor& min_val,
                        const reference_tests::Tensor& max_val,
                        ov::element::Type out_type,
                        int64_t global_seed,
                        int64_t op_seed,
                        const op::PhiloxAlignment alignment,
                        const reference_tests::Tensor& expected,
                        const std::string name)
        : out_shape(out_shape),
          min_val(min_val),
          max_val(max_val),
          out_type(out_type),
          global_seed(global_seed),
          op_seed(op_seed),
          alignment(alignment),
          expected(expected),
          test_case_name(std::move(name)) {}

    std::vector<int64_t> out_shape;
    reference_tests::Tensor min_val;
    reference_tests::Tensor max_val;
    ov::element::Type out_type;
    int64_t global_seed;
    int64_t op_seed;
    op::PhiloxAlignment alignment;
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
                                  params.op_seed,
                                  params.alignment);
        inputData = {params.min_val.data, params.max_val.data};
        refOutData = {params.expected.data};

        if (params.out_type == element::bf16) {
            const auto min = static_cast<bfloat16*>(params.min_val.data.data())[0];
            const auto max = static_cast<bfloat16*>(params.max_val.data.data())[0];
            if (max - min <= 1) {
                abs_threshold = 0.01f;  // Slight differences based on class implementation
            } else {
                abs_threshold = 1.01f;  // Differences in class implementation (rounding) can cause a difference of up
                                        // to 1 between values
            }
        } else {
            abs_threshold = 0.0f;  // Exact match
        }
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
                                                 int64_t op_seed,
                                                 const op::PhiloxAlignment alignment) {
        const auto out_shape_const =
            std::make_shared<op::v0::Constant>(element::i64, Shape{out_shape.size()}, out_shape);
        const auto min_val_param = std::make_shared<op::v0::Parameter>(min_val.type, min_val.shape);
        const auto max_val_param = std::make_shared<op::v0::Parameter>(max_val.type, max_val.shape);
        const auto random_uniform = std::make_shared<op::v8::RandomUniform>(out_shape_const,
                                                                            min_val_param,
                                                                            max_val_param,
                                                                            out_type,
                                                                            global_seed,
                                                                            op_seed,
                                                                            alignment);
        return std::make_shared<ov::Model>(random_uniform->outputs(), ParameterVector{min_val_param, max_val_param});
    }
};

TEST_P(ReferenceRandomUniformLayerTest, CompareWithRefs) {
    Exec();
}

}  // namespace

// Reference values for the following tests are obtained from single layer TensorFlow model with tf.random.uniform().
INSTANTIATE_TEST_SUITE_P(
    smoke_RandomUniform_With_Hardcoded_Refs,
    ReferenceRandomUniformLayerTest,
    ::testing::Values(
        // Backwards compatibility tests (implemented before alignment update)
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{0}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{1}},
                            element::Type_t::f32,
                            150,
                            10,
                            op::PhiloxAlignment::TENSORFLOW,
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
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {3, 2, 4},
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
                            op::PhiloxAlignment::TENSORFLOW,
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
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {3, 2, 4},
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
                            op::PhiloxAlignment::TENSORFLOW,
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
                            op::PhiloxAlignment::TENSORFLOW,
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
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 3},
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
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 3},
                                element::bf16,
                                std::vector<bfloat16>{164, 146, -92.5, -84.5, 14,  90,    33,  -43.5, 170, 8,  182,
                                                      -35, -24, -84.5, 180,   -14, -73.5, 198, 138,   8,   156}},
                            "bfloat16_non_default_min_max"),
        // Alignment tests - TENSORFLOW
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f64, std::vector<double>{0}},
                            reference_tests::Tensor{{1}, element::f64, std::vector<double>{1}},
                            element::f64,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f64,
                                std::vector<double>{
                                    0.2921047836089943,  0.1865942500848503,  0.006919873589045,   0.6934785411915267,
                                    0.24304703848961706, 0.03778989685165657, 0.6931826735517366,  0.24599033055460207,
                                    0.9268315651052301,  0.1651620289133835,  0.9018968607612903,  0.10734846627790451,
                                    0.9693998601811129,  0.32754183046231056, 0.6421559536150925,  0.5466934281951839,
                                    0.9409183236759846,  0.6341779150820659,  0.8806706943886118,  0.814950148009342,
                                    0.35454954890818113, 0.6674748438695257,  0.5283274225598105,  0.31927129954257216,
                                    0.9013092542329315,  0.8188783768209358,  0.9246842191826359,  0.08651125352955691,
                                    0.4156017869843349,  0.5489782764801512,  0.12346754160942996, 0.39029344640718455,
                                    0.6512753213561657,  0.2179250598578999,  0.8802885273401035,  0.016060267235235015,
                                    0.7303574589403252,  0.9344756372163827,  0.6340667403806175,  0.7051371357116014,
                                    0.5739520844965524,  0.1945947342973824}},
                            "double_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f64, std::vector<double>{-100}},
                            reference_tests::Tensor{{1}, element::f64, std::vector<double>{250}},
                            element::f64,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f64,
                                std::vector<double>{
                                    2.2366742631480037,  -34.69201247030239,  -97.57804424383426, 142.71748941703436,
                                    -14.933536528634022, -86.7735361019202,   142.6139357431078,  -13.903384305889276,
                                    224.39104778683054,  -42.193289880315774, 215.6639012664516,  -62.42803680273342,
                                    239.28995106338954,  14.639640661808698,  124.75458376528238, 91.34269986831436,
                                    229.32141328659463,  121.96227027872305,  208.23474303601415, 185.2325518032697,
                                    24.092342117863396,  133.616195354334,    84.91459789593367,  11.74495483990026,
                                    215.45823898152605,  186.60743188732755,  223.63947671392253, -69.72106126465508,
                                    45.460625444517234,  92.14239676805292,   -56.78636043669951, 36.602706242514586,
                                    127.946362474658,    -23.72622904973504,  208.10098456903626, -94.37890646766775,
                                    155.62511062911383,  227.06647302573396,  121.92335913321614, 146.7979974990605,
                                    100.88322957379336,  -31.891842995916164}},
                            "double_non_default_min_max_tensorflow"),
        RandomUniformParams(
            std::vector<int64_t>{7, 6},
            reference_tests::Tensor{{1}, element::f32, std::vector<float>{0}},
            reference_tests::Tensor{{1}, element::f32, std::vector<float>{1}},
            element::f32,
            12345,
            54321,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{
                {7, 6},
                element::f32,
                std::vector<float>{0.7865131,  0.5757234,  0.023324251, 0.21700966, 0.875865,   0.8446753,  0.2116847,
                                   0.8619245,  0.65538085, 0.21281981,  0.50472367, 0.3871348,  0.46164775, 0.1323191,
                                   0.9057487,  0.10803068, 0.3658539,   0.6284323,  0.89564514, 0.09049857, 0.61273706,
                                   0.16685092, 0.51341856, 0.99241984,  0.99617493, 0.02810657, 0.41594267, 0.23845005,
                                   0.83026946, 0.4635644,  0.3183366,   0.37955487, 0.99261475, 0.54943705, 0.20427215,
                                   0.64036727, 0.7350838,  0.8680873,   0.47686875, 0.1963104,  0.9193187,  0.6715238}},
            "float_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{-100}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{250}},
                            element::f32,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f32,
                                std::vector<float>{
                                    175.27957,  101.50319,  -91.83651,  -24.046616, 206.55273,  195.63635, -25.910355,
                                    201.67358,  129.3833,   -25.513062, 76.65329,   35.497177,  61.576706, -53.688316,
                                    217.01205,  -62.189262, 28.048874,  119.951294, 213.4758,   -68.3255,  114.45798,
                                    -41.602177, 79.69649,   247.34695,  248.66122,  -90.162704, 45.57994,  -16.54248,
                                    190.5943,   62.247543,  11.417809,  32.844208,  247.41516,  92.30296,  -28.504745,
                                    124.12854,  157.27933,  203.83057,  66.90407,   -31.291359, 221.76154, 135.03334}},
                            "float_non_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{0}},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{1}},
                            element::f16,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f16,
                                std::vector<float16>{0.11523, 0.3262, 0.07227, 0.743,  0.08594, 0.58,   0.1211,
                                                     0.8857,  0.88,   0.42,    0.6963, 0.4082,  0.8184, 0.958,
                                                     0.8936,  0.9873, 0.0752,  0.1172, 0.125,   0.3643, 0.542,
                                                     0.843,   0.925,  0.9033,  0.665,  0.249,   0.4023, 0.3828,
                                                     0.5674,  0.5195, 0.8135,  0.3135, 0.5,     0.9883, 0.3975,
                                                     0.8887,  0.8066, 0.371,   0.509,  0.1748,  0.0586, 0.12305}},
                            "float16_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{-100}},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{250}},
                            element::f16,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f16,
                                std::vector<float16>{-59.66, 14.19, -74.7, 160.0, -69.94, 103.0, -57.62, 210.0,  208.0,
                                                     47.0,   143.8, 42.88, 186.5, 235.2,  212.8, 245.5,  -73.7,  -59.0,
                                                     -56.25, 27.5,  89.75, 195.0, 223.8,  216.2, 132.8,  -12.81, 40.88,
                                                     34.0,   98.6,  81.9,  184.8, 9.69,   75.0,  246.0,  39.12,  211.0,
                                                     182.2,  29.88, 78.1,  -38.8, -79.5,  -56.94}},
                            "float16_non_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{0}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{1}},
                            element::bf16,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::bf16,
                                std::vector<bfloat16>{0.921875, 0.609375,  0.578125,  0.945312, 0.6875,   0.640625,
                                                      0.96875,  0.0859375, 0.0390625, 0.359375, 0.570312, 0.265625,
                                                      0.546875, 0.664062,  0.148438,  0.898438, 0.601562, 0.9375,
                                                      0,        0.914062,  0.335938,  0.742188, 0.398438, 0.226562,
                                                      0.320312, 0.992188,  0.21875,   0.0625,   0.539062, 0.15625,
                                                      0.507812, 0.507812,  0,         0.90625,  0.179688, 0.109375,
                                                      0.453125, 0.96875,   0.0703125, 0.398438, 0.46875,  0.984375}},
                            "bfloat16_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{-100}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{250}},
                            element::bf16,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::bf16,
                                std::vector<bfloat16>{222,  113, 102,    230, 141, 124,   240, -70, -86.5,
                                                      26,   100, -7,     91,  132, -48,   214, 111, 228,
                                                      -100, 220, 17.5,   160, 39,  -20.5, 12,  248, -23.5,
                                                      -78,  89,  -45.25, 78,  78,  -100,  218, -37, -61.75,
                                                      59,   240, -75.5,  39,  64,  244}},
                            "bfloat16_non_default_min_max_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{-100}},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{250}},
                            element::i64,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::i64,
                                std::vector<int64_t>{230, -4,  26,  -12, 23,  195, 74,  -53, 121, 206, -99,
                                                     9,   249, -70, -59, 101, 52,  -41, 210, 225, 76,  -54,
                                                     183, -12, 237, -60, -21, 159, 134, 147, 37,  116, -13,
                                                     100, 53,  182, 143, 228, -57, 93,  230, 195}},
                            "int64_tensorflow"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-100}},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{250}},
                            element::i32,
                            12345,
                            54321,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{
                                {7, 6},
                                element::i32,
                                std::vector<int32_t>{-92, 82,  160, 41,  218, 48,  42,  51,  189, -96, -95,
                                                     -10, -72, 101, 175, 207, 31,  -60, 126, -45, 131, 245,
                                                     77,  -33, 157, 77,  60,  -80, 79,  72,  -49, 25,  -70,
                                                     32,  -3,  -78, 82,  -82, -71, 51,  -24, 100}},
                            "int32_non_default_min_max_tensorflow"),

        // Alignment tests - PyTorch
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f64, std::vector<double>{0}},
                            reference_tests::Tensor{{1}, element::f64, std::vector<double>{1}},
                            element::f64,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f64,
                                std::vector<double>{
                                    0.8537458849690976,  0.9371476796661825,  0.6657276343152506,  0.9394525654011401,
                                    0.7008540783300001,  0.6755513936567185,  0.32574008079829586, 0.7066936129275108,
                                    0.7607924337011228,  0.5111041808001306,  0.3199238263815529,  0.7561993726681363,
                                    0.17923607421035226, 0.9980856889457874,  0.7451967274135151,  0.33009892801967744,
                                    0.49610515018696655, 0.21627452678551762, 0.5982817840191142,  0.10759022259137951,
                                    0.789470580212996,   0.34500166455063475, 0.6427628345630576,  0.7974227385272604,
                                    0.3921580931256149,  0.4032344020301808,  0.741876277138582,   0.2373560606013575,
                                    0.6045086845253197,  0.03240364251463124, 0.9873074507572961,  0.15825495647484955,
                                    0.8237589940447471,  0.5368713853196583,  0.09850875468220899, 0.7705821561611728,
                                    0.35758637185744746, 0.40904791646937666, 0.143346220011503,   0.4891462815971308,
                                    0.9808312595026292,  0.4561964890218795}},
                            "double_default_min_max_pytorch"),
        RandomUniformParams(
            std::vector<int64_t>{7, 6},
            reference_tests::Tensor{{1}, element::f64, std::vector<double>{-100}},
            reference_tests::Tensor{{1}, element::f64, std::vector<double>{250}},
            element::f64,
            12345,
            54321,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{
                {7, 6},
                element::f64,
                std::vector<double>{
                    198.81105973918417, 228.00168788316387, 133.0046720103377,   228.80839789039902, 145.29892741550003,
                    136.44298777985145, 14.009028279403552, 147.34276452462876,  166.27735179539297, 78.88646328004573,
                    11.973339233543523, 164.6697804338477,  -37.26737402637671,  249.3299911310256,  160.81885459473028,
                    15.534624806887104, 73.63680256543829,  -24.303915625068832, 109.39862440668999, -62.34342209301717,
                    176.3147030745486,  20.75058259272216,  124.96699209707016,  179.09795848454115, 37.255332593965214,
                    41.13204071056328,  159.6566969985037,  -16.925378789524874, 111.5780395838619,  -88.65872511987907,
                    245.55760776505363, -44.61076523380266, 188.31564791566146,  87.9049848618804,   -65.52193586122685,
                    169.70375465641047, 25.15523015010661,  43.16677076428183,   -49.82882299597395, 71.20119855899578,
                    243.29094082592022, 59.668771157657815}},
            "double_non_default_min_max_pytorch"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{0}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{1}},
                            element::f32,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f32,
                                std::vector<float>{
                                    0.9817181825637817,  0.8796065449714661,  0.992143452167511,   0.461067259311676,
                                    0.08321595191955566, 0.17843109369277954, 0.36743152141571045, 0.5676497220993042,
                                    0.3376067280769348,  0.21194660663604736, 0.4594438672065735,  0.8153534531593323,
                                    0.9157174825668335,  0.2531347870826721,  0.21333664655685425, 0.4769676923751831,
                                    0.7200990319252014,  0.7238213419914246,  0.3138880133628845,  0.673179030418396,
                                    0.4149904251098633,  0.4399939775466919,  0.09452491998672485, 0.858170211315155,
                                    0.1474044919013977,  0.624611496925354,   0.2497606873512268,  0.07847321033477783,
                                    0.4681495428085327,  0.6659092307090759,  0.04126232862472534, 0.5361465811729431,
                                    0.31201308965682983, 0.4287737011909485,  0.27703428268432617, 0.4377092719078064,
                                    0.9497851729393005,  0.01932889223098755, 0.2634487748146057,  0.9249169230461121,
                                    0.47368377447128296, 0.3961203098297119}},
                            "float_default_min_max_pytorch"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{-100}},
                            reference_tests::Tensor{{1}, element::f32, std::vector<float>{250}},
                            element::f32,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f32,
                                std::vector<float>{
                                    243.6013641357422,  207.86228942871094,  247.25021362304688,  61.373538970947266,
                                    -70.87442016601562, -37.54911804199219,  28.601032257080078,  98.67740631103516,
                                    18.162355422973633, -25.818687438964844, 60.805355072021484,  185.3737030029297,
                                    220.50111389160156, -11.402824401855469, -25.33217430114746,  66.93869018554688,
                                    152.03466796875,    153.33746337890625,  9.860804557800293,   135.61265563964844,
                                    45.24665069580078,  53.99789047241211,   -66.91627502441406,  200.3595733642578,
                                    -48.40842819213867, 118.61402130126953,  -12.583759307861328, -72.53437805175781,
                                    63.85234069824219,  133.0682373046875,   -85.55818176269531,  87.65130615234375,
                                    9.204581260681152,  50.070796966552734,  -3.03800106048584,   53.198246002197266,
                                    232.4248046875,     -93.2348861694336,   -7.792928695678711,  223.72091674804688,
                                    65.78932189941406,  38.64210891723633}},
                            "float_non_default_min_max_pytorch"),
        RandomUniformParams(
            std::vector<int64_t>{7, 6},
            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{0}},
            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{1}},
            element::f16,
            12345,
            54321,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{
                {7, 6},
                element::f16,
                std::vector<float16>{
                    0.2353515625,  0.73681640625, 0.63916015625, 0.06298828125, 0.705078125,   0.70751953125,
                    0.9990234375,  0.1865234375,  0.67431640625, 0.2666015625,  0.76416015625, 0.37548828125,
                    0.5576171875,  0.68017578125, 0.65380859375, 0.3193359375,  0.05126953125, 0.54443359375,
                    0.37060546875, 0.6826171875,  0.6015625,     0.4306640625,  0.34814453125, 0.13037109375,
                    0.53759765625, 0.8173828125,  0.03955078125, 0.8525390625,  0.0810546875,  0.12841796875,
                    0.02099609375, 0.11279296875, 0.01123046875, 0.51416015625, 0.46484375,    0.71435546875,
                    0.64013671875, 0.34228515625, 0.17236328125, 0.91943359375, 0.41748046875, 0.017578125}},
            "float16_default_min_max_pytorch"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{-100}},
                            reference_tests::Tensor{{1}, element::f16, std::vector<float16>{250}},
                            element::f16,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::f16,
                                std::vector<float16>{-17.625,  157.875,   123.6875, -77.9375,  146.75,   147.625,
                                                     249.625,  -34.71875, 136.0,    -6.6875,   167.5,    31.421875,
                                                     95.1875,  138.0,     128.875,  11.765625, -82.0625, 90.5625,
                                                     29.71875, 138.875,   110.5625, 50.71875,  21.84375, -54.375,
                                                     88.1875,  186.125,   -86.1875, 198.375,   -71.625,  -55.0625,
                                                     -92.625,  -60.53125, -96.0625, 79.9375,   62.6875,  150.0,
                                                     124.0625, 19.796875, -39.6875, 221.75,    46.125,   -93.875}},
                            "float16_non_default_min_max_pytorch"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{0}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{1}},
                            element::bf16,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::bf16,
                                std::vector<bfloat16>{
                                    0.8828125,  0.89453125, 0.11328125, 0.50390625, 0.640625,   0.66015625, 0.9921875,
                                    0.4921875,  0.39453125, 0.1328125,  0.11328125, 0.00390625, 0.4609375,  0.44140625,
                                    0.23046875, 0.5546875,  0.41015625, 0.35546875, 0.96484375, 0.4609375,  0.8125,
                                    0.4453125,  0.78515625, 0.04296875, 0.30078125, 0.5390625,  0.31640625, 0.8203125,
                                    0.6484375,  0.02734375, 0.16796875, 0.90234375, 0.08984375, 0.11328125, 0.71875,
                                    0.71484375, 0.12109375, 0.73828125, 0.37890625, 0.35546875, 0.33984375, 0.140625}},
                            "bfloat16_default_min_max_pytorch"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{-100}},
                            reference_tests::Tensor{{1}, element::bf16, std::vector<bfloat16>{250}},
                            element::bf16,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::bf16,
                                std::vector<bfloat16>{
                                    209.0,  213.0, -60.25,  76.5,    124.0, 131.0, 247.0,  72.5,  38.0,   -53.5, -60.25,
                                    -98.5,  61.25, 54.5,    -19.375, 94.0,  43.5,  24.375, 238.0, 61.25,  184.0, 55.75,
                                    175.0,  -85.0, 5.28125, 88.5,    10.75, 187.0, 127.0,  -90.5, -41.25, 216.0, -68.5,
                                    -60.25, 152.0, 150.0,   -57.5,   158.0, 32.5,  24.375, 19.0,  -50.75}},
                            "bfloat16_non_default_min_max_pytorch"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{-100}},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{250}},
                            element::i64,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::i64,
                                std::vector<int64_t>{90,  181, 85,  -97, 34,  -63, 160, 158, -79, 50,  121,
                                                     -35, 226, 85,  -83, 66,  13,  -73, 239, 244, 240, 222,
                                                     -49, 153, 119, 76,  121, 144, 12,  -1,  105, 107, -27,
                                                     141, 232, 165, 237, 125, 47,  123, -81, 242}},
                            "int64_pytorch"),
        RandomUniformParams(std::vector<int64_t>{2, 4},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{0}},
                            reference_tests::Tensor{{1}, element::i64, std::vector<int64_t>{4294967300}},
                            element::i64,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{{2, 4},
                                                    element::i64,
                                                    std::vector<int64_t>{737404521,
                                                                         3716027413,
                                                                         1306031901,
                                                                         35197318,
                                                                         1121750166,
                                                                         2465874581,
                                                                         2593594281,
                                                                         1335862698}},
                            "int64_pytorch_high_values"),
        RandomUniformParams(std::vector<int64_t>{7, 6},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{-100}},
                            reference_tests::Tensor{{1}, element::i32, std::vector<int32_t>{250}},
                            element::i32,
                            12345,
                            54321,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{
                                {7, 6},
                                element::i32,
                                std::vector<int32_t>{90,  181, 85,  -97, 34,  -63, 160, 158, -79, 50,  121,
                                                     -35, 226, 85,  -83, 66,  13,  -73, 239, 244, 240, 222,
                                                     -49, 153, 119, 76,  121, 144, 12,  -1,  105, 107, -27,
                                                     141, 232, 165, 237, 125, 47,  123, -81, 242}},
                            "int32_pytorch")),
    ReferenceRandomUniformLayerTest::getTestCaseName);
}  // namespace reference_tests
