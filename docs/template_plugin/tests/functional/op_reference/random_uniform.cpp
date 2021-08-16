// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "base_reference_test.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

namespace reference_tests {
namespace {

struct RandomUniformParams {
    RandomUniformParams(const std::vector<int64_t>& paramOutShape,
                        const Tensor& paramMinValue,
                        const Tensor& paramMaxValue,
                        ngraph::element::Type paramOutType,
                        int64_t paramGlobalSeed,
                        int64_t paramOpSeed,
                        const Tensor& paramExpected,
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
    Tensor min_val;
    Tensor max_val;
    ngraph::element::Type out_type;
    int64_t global_seed;
    int64_t op_seed;
    Tensor expected;
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
    static std::shared_ptr<Function> CreateFunction(const std::vector<int64_t>& out_shape,
                                                    const Tensor& min_val,
                                                    const Tensor& max_val,
                                                    ngraph::element::Type out_type,
                                                    int64_t global_seed,
                                                    int64_t op_seed) {
        const auto min_val_param = std::make_shared<opset8::Parameter>(min_val.type, min_val.shape);
        const auto max_val_param = std::make_shared<opset8::Parameter>(max_val.type, max_val.shape);
        auto out_shape_ = std::make_shared<opset8::Constant>(element::i64, Shape{out_shape.size()}, out_shape);

        return std::make_shared<Function>(NodeVector{std::make_shared<opset8::RandomUniform>(out_shape_,
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

INSTANTIATE_TEST_SUITE_P(
    smoke_RandomUniform_With_Hardcoded_Refs,
    ReferenceRandomUniformLayerTest,
    ::testing::Values(
        RandomUniformParams(
            std::vector<int64_t>{3, 2, 4},
            Tensor{{1}, element::f32, std::vector<float>{0}},
            Tensor{{1}, element::f32, std::vector<float>{1}},
            element::Type_t::f32,
            150,
            10,
            Tensor{{3, 2, 4},
                   element::f32,
                   std::vector<float>{
                       0.7011235952377319336, 0.3053963184356689453, 0.9393105506896972656, 0.9456034898757934570,
                       0.1169477701187133789, 0.5077005624771118164, 0.5197197198867797852, 0.2272746562957763672,
                       0.9913740158081054688, 0.3551903963088989258, 0.8269231319427490234, 0.5986485481262207031,
                       0.3136410713195800781, 0.5748131275177001953, 0.4139908552169799805, 0.9630825519561767578,
                       0.3714079856872558594, 0.8525316715240478516, 0.0935858488082885742, 0.0820095539093017578,
                       0.2365508079528808594, 0.8105630874633789062, 0.7422660589218139648, 0.7610669136047363281}},
            "float32_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            Tensor{{1}, element::f16, std::vector<float16>{0}},
                            Tensor{{1}, element::f16, std::vector<float16>{1}},
                            element::Type_t::f16,
                            150,
                            10,
                            Tensor{{3, 2, 4},
                                   element::f16,
                                   std::vector<float16>{0.6044921875, 0.8066406250, 0.8320312500, 0.3837890625,
                                                        0.0361328125, 0.0830078125, 0.5439453125, 0.8339843750,
                                                        0.3359375000, 0.7197265625, 0.1542968750, 0.1289062500,
                                                        0.3476562500, 0.8691406250, 0.4130859375, 0.5722656250,
                                                        0.5742187500, 0.9394531250, 0.6552734375, 0.8222656250,
                                                        0.8242187500, 0.1328125000, 0.6435546875, 0.6601562500}},
                            "float16_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            Tensor{{1}, element::f32, std::vector<float>{-650}},
                            Tensor{{1}, element::f32, std::vector<float>{450}},
                            element::Type_t::f32,
                            150,
                            10,
                            Tensor{{3, 2, 4},
                                   element::f32,
                                   std::vector<float>{
                                       121.2359619140625000000,  -314.0640563964843750000, 383.2415771484375000000,
                                       390.1638183593750000000,  -521.3574218750000000000, -91.5293579101562500000,
                                       -78.3082885742187500000,  -399.9978637695312500000, 440.5114746093750000000,
                                       -259.2905578613281250000, 259.6154174804687500000,  8.5134277343750000000,
                                       -304.9948120117187500000, -17.7055664062500000000,  -194.6100463867187500000,
                                       409.3907470703125000000,  -241.4512023925781250000, 287.7848510742187500000,
                                       -547.0555419921875000000, -559.7894897460937500000, -389.7940979003906250000,
                                       241.6193847656250000000,  166.4926757812500000000,  187.1735839843750000000}},
                            "float32_non_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{3, 2, 4},
                            Tensor{{1}, element::f16, std::vector<float16>{-1.5}},
                            Tensor{{1}, element::f16, std::vector<float16>{-1.0}},
                            element::Type_t::f16,
                            150,
                            10,
                            Tensor{{3, 2, 4},
                                   element::f16,
                                   std::vector<float16>{-1.1972656250, -1.0966796875, -1.0839843750, -1.3085937500,
                                                        -1.4824218750, -1.4589843750, -1.2285156250, -1.0830078125,
                                                        -1.3320312500, -1.1406250000, -1.4228515625, -1.4355468750,
                                                        -1.3261718750, -1.0654296875, -1.2929687500, -1.2138671875,
                                                        -1.2128906250, -1.0302734375, -1.1718750000, -1.0888671875,
                                                        -1.0878906250, -1.4335937500, -1.1777343750, -1.1699218750}},
                            "float16_non_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{2, 3, 4},
                            Tensor{{1}, element::i32, std::vector<int32_t>{-100}},
                            Tensor{{1}, element::i32, std::vector<int32_t>{50}},
                            element::Type_t::i32,
                            100,
                            350,
                            Tensor{{2, 3, 4},
                                   element::i32,
                                   std::vector<int32_t>{
                                       22, -56, -33, -89, -98, -33, -3,  -48, -82, 5,  -66, 21,
                                       29, -42, -73, -37, 3,   36,  -35, 20,  -11, -8, -78, 47,
                                   }},
                            "int32"),
        RandomUniformParams(std::vector<int64_t>{5, 4, 3},
                            Tensor{{1}, element::i64, std::vector<int64_t>{-2600}},
                            Tensor{{1}, element::i64, std::vector<int64_t>{3700}},
                            element::Type_t::i64,
                            755,
                            951,
                            Tensor{{5, 4, 3},
                                   element::i64,
                                   std::vector<int64_t>{
                                       2116, -1581, 2559,  -339,  -1660, 519,   90,   2027,  -210, 3330, 1831,  -1737,
                                       2683, 2661,  3473,  1220,  3534,  -2384, 2199, 1935,  499,  2861, 2743,  3223,
                                       -531, -836,  -65,   3435,  632,   1765,  2613, 1891,  1698, 3069, 169,   -792,
                                       -32,  2976,  -1552, -2588, 3327,  -1756, 2637, -1084, 3567, -778, -1465, 2967,
                                       1242, 2672,  -1585, -2271, 3536,  -1502, 400,  2241,  3126, 908,  1073,  -2110}},
                            "int64"),
        RandomUniformParams(std::vector<int64_t>{7, 3},
                            Tensor{{1}, element::bf16, std::vector<bfloat16>{0}},
                            Tensor{{1}, element::bf16, std::vector<bfloat16>{1}},
                            element::Type_t::bf16,
                            4978,
                            5164,
                            Tensor{{7, 3},
                                   element::bf16,
                                   std::vector<bfloat16>{0.8984375, 0.84375,   0.1640625, 0.1875,   0.46875,  0.6875,
                                                         0.5234375, 0.3046875, 0.9140625, 0.453125, 0.953125, 0.328125,
                                                         0.359375,  0.1875,    0.9453125, 0.390625, 0.21875,  0.9921875,
                                                         0.8203125, 0.453125,  0.875}},
                            "bfloat16_default_min_max"),
        RandomUniformParams(std::vector<int64_t>{7, 3},
                            Tensor{{1}, element::bf16, std::vector<bfloat16>{-150}},
                            Tensor{{1}, element::bf16, std::vector<bfloat16>{200}},
                            element::Type_t::bf16,
                            4978,
                            5164,
                            Tensor{{7, 3},
                                   element::bf16,
                                   std::vector<bfloat16>{164, 146, -92.5, -84.5, 14,  90,    33,  -43.5, 170, 8,  182,
                                                         -35, -24, -84.5, 180,   -14, -73.5, 198, 138,   8,   156}},
                            "bfloat16_non_default_min_max")),
    ReferenceRandomUniformLayerTest::getTestCaseName);
}  // namespace reference_tests
