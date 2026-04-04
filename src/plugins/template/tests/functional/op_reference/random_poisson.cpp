// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/random_poisson.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;

namespace reference_tests {
namespace {

// Define the Parameters for the RandomPoisson test
struct RandomPoissonParams {
    RandomPoissonParams(const reference_tests::Tensor& input,
                        uint64_t global_seed,
                        uint64_t op_seed,
                        op::PhiloxAlignment alignment,
                        const reference_tests::Tensor& expected,
                        const std::string name)
        : input(input),
          global_seed(global_seed),
          op_seed(op_seed),
          alignment(alignment),
          expected(expected),
          test_case_name(std::move(name)) {}

    reference_tests::Tensor input;
    uint64_t global_seed;
    int64_t op_seed;
    op::PhiloxAlignment alignment;
    reference_tests::Tensor expected;
    std::string test_case_name;
};

class ReferenceRandomPoissonLayerTest : public testing::TestWithParam<RandomPoissonParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input, params.global_seed, params.op_seed, params.alignment);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
        // We can set the threshold if needed but there no need for now
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RandomPoissonParams>& obj) {
        auto param = obj.param;
        return param.test_case_name;
    }

private:
    static std::shared_ptr<Model> CreateFunction(const reference_tests::Tensor& input,
                                                 uint64_t global_seed,
                                                 uint64_t op_seed,
                                                 ov::op::PhiloxAlignment alignment) {
        auto input_param = std::make_shared<op::v0::Parameter>(input.type, input.shape);
        auto random_poisson = std::make_shared<op::v17::RandomPoisson>(input_param, global_seed, op_seed, alignment);
        return std::make_shared<ov::Model>(random_poisson->outputs(), ParameterVector{input_param});
    }
};

TEST_P(ReferenceRandomPoissonLayerTest, CompareWithRefs) {
    Exec();
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_RandomPoisson_With_Hardcoded_Refs,
    ReferenceRandomPoissonLayerTest,
    ::testing::Values(
        // Alignment tests - PyTorch
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{1.0, 2.0, 2.0, 2.0, 3.0, 8.0, 9.0, 11.0, 5.0}},
            "bfloat16_pytorch_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{14.0, 11.0, 15.0, 20.0, 9.0, 20.0, 15.0, 17.0, 21.0}},
            "bfloat16_pytorch_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{0.0, 6.0, 6.0, 11.0, 19.0, 15.0, 36.0, 33.0, 33.0}},
            "bfloat16_pytorch_mixed_knuth_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::bf16,
                                    std::vector<ov::bfloat16>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            "bfloat16_pytorch_all_0"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{1.0, 2.0, 2.0, 2.0, 3.0, 8.0, 9.0, 11.0, 5.0}},
            "float16_pytorch_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{14.0, 11.0, 15.0, 20.0, 9.0, 20.0, 15.0, 17.0, 21.0}},
            "float16_pytorch_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 6.0, 6.0, 11.0, 19.0, 15.0, 36.0, 33.0, 33.0}},
            "float16_pytorch_mixed_knuth_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            "float16_pytorch_all_0"),
        RandomPoissonParams(
            reference_tests::Tensor{{9}, element::f32, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 8.0f, 9.0f, 11.0f, 5.0f}},
            "float32_pytorch_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{14.0f, 11.0f, 15.0f, 20.0f, 9.0f, 20.0f, 15.0f, 17.0f, 21.0f}},
            "float32_pytorch_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 6.0f, 6.0f, 11.0f, 19.0f, 15.0f, 36.0f, 33.0f, 33.0f}},
            "float32_pytorch_mixed_knuth_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
            "float32_pytorch_all_0"),
        RandomPoissonParams(reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
                            150,
                            69,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{1.0, 2.0, 2.0, 2.0, 3.0, 8.0, 9.0, 11.0, 5.0}},
                            "float64_pytorch_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{14.0, 11.0, 15.0, 20.0, 9.0, 20.0, 15.0, 17.0, 21.0}},
            "float64_pytorch_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0}},
            150,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{0.0, 6.0, 6.0, 11.0, 19.0, 15.0, 36.0, 33.0, 33.0}},
            "float64_pytorch_mixed_knuth_hoermann"),
        RandomPoissonParams(reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                            150,
                            69,
                            op::PhiloxAlignment::PYTORCH,
                            reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                            "float64_pytorch_all_0"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {7},
                element::f64,
                std::vector<double>{78.59323, 55.1172, 24.38882, 33.44532, 31.86902, 39.0154, 80.13007}},
            77,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{7}, element::f64, std::vector<double>{74.0, 57.0, 36.0, 30.0, 32.0, 39.0, 62.0}},
            "float64_pytorch_mixed_knuth_hoermann_1d_tensor"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {7, 11},
                element::f64,
                std::vector<double>{
                    0.4822044, 93.0787,  78.45822, 33.4325,  68.62689, 73.22626, 82.46472, 10.74544, 30.89917, 31.29434,
                    59.40727,  61.05506, 91.65421, 73.88172, 37.74708, 79.49533, 48.69355, 57.88914, 97.44749, 68.83897,
                    58.48087,  27.39643, 83.77858, 30.22294, 17.28462, 80.17434, 34.41099, 43.75847, 15.26812, 27.61358,
                    56.38672,  86.46747, 8.27886,  97.48236, 92.28236, 40.73737, 78.07689, 64.65956, 22.54609, 82.44807,
                    34.53563,  56.85671, 68.63697, 24.70088, 87.53744, 15.59331, 69.52023, 16.52205, 48.09663, 52.20325,
                    57.47133,  84.15278, 48.84156, 91.6499,  12.45573, 45.37853, 95.8913,  67.06483, 79.28513, 76.38069,
                    46.94303,  26.84749, 45.32233, 25.27348, 94.72152, 4.974078, 43.75196, 76.8388,  16.66324, 82.44444,
                    86.96023,  44.59329, 16.04106, 41.98496, 45.22867, 85.44545, 74.39886}},
            77,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{
                {7, 11},
                element::f64,
                std::vector<double>{0.0,  84.0,  73.0, 25.0, 74.0, 65.0,  83.0, 10.0, 40.0, 29.0, 60.0, 55.0, 85.0,
                                    59.0, 33.0,  95.0, 48.0, 48.0, 108.0, 71.0, 63.0, 22.0, 92.0, 28.0, 15.0, 80.0,
                                    37.0, 50.0,  12.0, 35.0, 37.0, 86.0,  6.0,  95.0, 74.0, 49.0, 68.0, 65.0, 23.0,
                                    70.0, 25.0,  75.0, 70.0, 27.0, 69.0,  14.0, 69.0, 23.0, 55.0, 60.0, 55.0, 78.0,
                                    61.0, 109.0, 13.0, 51.0, 94.0, 50.0,  80.0, 82.0, 52.0, 26.0, 46.0, 27.0, 104.0,
                                    6.0,  38.0,  89.0, 14.0, 77.0, 79.0,  47.0, 15.0, 52.0, 48.0, 81.0, 79.0}},
            "float64_pytorch_mixed_knuth_hoermann_2d_tensor"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {2, 7, 2},
                element::f64,
                std::vector<double>{
                    58.03724, 52.6852,  86.49617, 73.47604, 47.65848, 81.66962, 54.46348, 43.74158, 0.8451105, 67.84737,
                    78.63136, 70.59277, 96.21093, 72.16828, 1.143179, 92.75036, 82.25292, 68.17147, 73.3827,   6.819783,
                    1.821376, 53.76863, 90.18542, 72.11462, 89.51485, 29.34519, 46.98525, 28.66298}},
            77,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{{2, 7, 2},
                                    element::f64,
                                    std::vector<double>{54.0, 55.0, 109.0, 69.0, 48.0, 82.0, 40.0, 41.0, 0.0,  65.0,
                                                        80.0, 64.0, 90.0,  58.0, 0.0,  91.0, 82.0, 76.0, 72.0, 8.0,
                                                        2.0,  71.0, 88.0,  74.0, 80.0, 31.0, 51.0, 45.0}},
            "float64_pytorch_mixed_knuth_hoermann_3d_tensor"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {3, 4, 1, 7},
                element::f64,
                std::vector<double>{
                    62.50955, 89.72138,  77.56857, 22.52072, 30.01663, 87.35534, 0.5265305, 82.12284, 79.70694,
                    46.7935,  30.30324,  27.84256, 25.48696, 44.50763, 50.45483, 55.34974,  99.55003, 79.26619,
                    62.21792, 98.89601,  21.53087, 16.0212,  61.25396, 4.394201, 3.568028,  51.48888, 46.6206,
                    91.71678, 62.92263,  51.41176, 49.68734, 24.75149, 1.179403, 19.24021,  69.20321, 20.06067,
                    36.95363, 0.3734242, 83.00477, 15.44611, 26.75993, 88.03322, 50.97908,  84.71502, 63.97172,
                    74.17709, 9.149561,  54.11438, 50.77722, 87.13394, 36.12641, 59.81841,  5.925164, 38.76318,
                    32.30363, 15.01997,  81.63381, 37.94462, 97.87479, 58.99917, 60.50563,  63.79966, 67.64502,
                    15.0788,  44.03135,  23.9564,  40.24983, 9.670409, 96.78281, 21.5004,   67.17652, 30.04201,
                    87.4077,  66.22147,  13.16158, 84.50743, 94.49482, 90.39168, 56.97191,  14.546,   19.24635,
                    92.79057, 55.23265,  18.05525}},
            77,
            69,
            op::PhiloxAlignment::PYTORCH,
            reference_tests::Tensor{
                {3, 4, 1, 7},
                element::f64,
                std::vector<double>{59.0, 92.0,  87.0, 34.0,  27.0,  88.0, 0.0,  83.0,  79.0, 58.0,  28.0,  29.0,
                                    21.0, 40.0,  38.0, 49.0,  117.0, 78.0, 52.0, 109.0, 23.0, 19.0,  54.0,  6.0,
                                    2.0,  58.0,  41.0, 105.0, 42.0,  51.0, 48.0, 21.0,  2.0,  18.0,  54.0,  15.0,
                                    38.0, 0.0,   79.0, 12.0,  26.0,  75.0, 52.0, 102.0, 63.0, 80.0,  9.0,   59.0,
                                    73.0, 102.0, 47.0, 56.0,  10.0,  26.0, 33.0, 18.0,  88.0, 37.0,  100.0, 61.0,
                                    68.0, 75.0,  63.0, 12.0,  41.0,  15.0, 44.0, 6.0,   89.0, 23.0,  64.0,  38.0,
                                    91.0, 62.0,  15.0, 91.0,  81.0,  97.0, 42.0, 23.0,  23.0, 110.0, 59.0,  18.0}},
            "float64_pytorch_mixed_knuth_hoermann_4d_tensor"),
        // Alignment tests - TensorFlow
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{1.0, 1.0, 1.0, 2.0, 9.0, 5.0, 3.0, 3.0, 8.0}},
            "float16_tensorflow_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{12.0, 8.0, 7.0, 11.0, 15.0, 12.0, 23.0, 21.0, 15.0}},
            "float16_tensorflow_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 4.0, 5.0, 12.0, 21.0, 20.0, 38.0, 39.0, 34.0}},
            "float16_tensorflow_mixed_knuth_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f16,
                                    std::vector<ov::float16>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
            "float16_tensorflow_all_0"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{1.0f, 1.0f, 1.0f, 2.0f, 9.0f, 5.0f, 3.0f, 3.0f, 8.0f}},
            "float32_tensorflow_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{12.0f, 8.0f, 7.0f, 11.0f, 15.0f, 12.0f, 23.0f, 21.0f, 15.0f}},
            "float32_tensorflow_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f, 30.0f, 35.0f, 40.0f}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 4.0f, 5.0f, 12.0f, 21.0f, 20.0f, 38.0f, 39.0f, 34.0f}},
            "float32_tensorflow_mixed_knuth_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f32,
                                    std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
            "float32_tensorflow_all_0"),
        RandomPoissonParams(reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}},
                            150,
                            77,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{1.0, 1.0, 1.0, 2.0, 9.0, 5.0, 3.0, 3.0, 8.0}},
                            "float64_tensorflow_knuth"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{12.0, 8.0, 7.0, 11.0, 15.0, 12.0, 23.0, 21.0, 15.0}},
            "float64_tensorflow_hoermann"),
        RandomPoissonParams(
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0}},
            150,
            77,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{9},
                                    element::f64,
                                    std::vector<double>{0.0, 4.0, 5.0, 12.0, 21.0, 20.0, 38.0, 39.0, 34.0}},
            "float64_tensorflow_mixed_knuth_hoermann"),
        RandomPoissonParams(reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                            150,
                            77,
                            op::PhiloxAlignment::TENSORFLOW,
                            reference_tests::Tensor{{9},
                                                    element::f64,
                                                    std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                            "float64_tensorflow_all_0"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {7},
                element::f64,
                std::vector<double>{78.59323, 55.1172, 24.38882, 33.44532, 31.86902, 39.0154, 80.13007}},
            77,
            69,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{7}, element::f64, std::vector<double>{84.0, 57.0, 23.0, 29.0, 21.0, 24.0, 82.0}},
            "float64_tensorflow_mixed_knuth_hoermann_1d_tensor"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {7, 11},
                element::f64,
                std::vector<double>{
                    0.4822044, 93.0787,  78.45822, 33.4325,  68.62689, 73.22626, 82.46472, 10.74544, 30.89917, 31.29434,
                    59.40727,  61.05506, 91.65421, 73.88172, 37.74708, 79.49533, 48.69355, 57.88914, 97.44749, 68.83897,
                    58.48087,  27.39643, 83.77858, 30.22294, 17.28462, 80.17434, 34.41099, 43.75847, 15.26812, 27.61358,
                    56.38672,  86.46747, 8.27886,  97.48236, 92.28236, 40.73737, 78.07689, 64.65956, 22.54609, 82.44807,
                    34.53563,  56.85671, 68.63697, 24.70088, 87.53744, 15.59331, 69.52023, 16.52205, 48.09663, 52.20325,
                    57.47133,  84.15278, 48.84156, 91.6499,  12.45573, 45.37853, 95.8913,  67.06483, 79.28513, 76.38069,
                    46.94303,  26.84749, 45.32233, 25.27348, 94.72152, 4.974078, 43.75196, 76.8388,  16.66324, 82.44444,
                    86.96023,  44.59329, 16.04106, 41.98496, 45.22867, 85.44545, 74.39886}},
            77,
            69,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{
                {7, 11},
                element::f64,
                std::vector<double>{1.0,  96.0,  77.0, 29.0, 53.0,  53.0,  84.0, 12.0, 30.0, 26.0, 67.0,  49.0, 105.0,
                                    78.0, 26.0,  94.0, 48.0, 45.0,  104.0, 59.0, 54.0, 17.0, 79.0, 38.0,  21.0, 78.0,
                                    30.0, 36.0,  19.0, 31.0, 55.0,  79.0,  11.0, 93.0, 89.0, 54.0, 83.0,  51.0, 13.0,
                                    70.0, 25.0,  48.0, 68.0, 20.0,  77.0,  11.0, 72.0, 14.0, 42.0, 48.0,  47.0, 92.0,
                                    46.0, 104.0, 14.0, 42.0, 103.0, 69.0,  82.0, 73.0, 50.0, 26.0, 40.0,  27.0, 86.0,
                                    9.0,  48.0,  62.0, 15.0, 97.0,  86.0,  40.0, 15.0, 48.0, 43.0, 105.0, 73.0}},
            "float64_tensorflow_mixed_knuth_hoermann_2d_tensor"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {2, 7, 2},
                element::f64,
                std::vector<double>{
                    58.03724, 52.6852,  86.49617, 73.47604, 47.65848, 81.66962, 54.46348, 43.74158, 0.8451105, 67.84737,
                    78.63136, 70.59277, 96.21093, 72.16828, 1.143179, 92.75036, 82.25292, 68.17147, 73.3827,   6.819783,
                    1.821376, 53.76863, 90.18542, 72.11462, 89.51485, 29.34519, 46.98525, 28.66298}},
            77,
            69,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{{2, 7, 2},
                                    element::f64,
                                    std::vector<double>{63.0, 55.0, 85.0,  67.0, 34.0, 60.0,  56.0, 46.0, 1.0,  61.0,
                                                        87.0, 58.0, 106.0, 76.0, 0.0,  109.0, 82.0, 54.0, 79.0, 12.0,
                                                        2.0,  75.0, 86.0,  84.0, 98.0, 28.0,  42.0, 23.0}},
            "float64_tensorflow_mixed_knuth_hoermann_3d_tensor"),
        RandomPoissonParams(
            reference_tests::Tensor{
                {3, 4, 1, 7},
                element::f64,
                std::vector<double>{
                    62.50955, 89.72138,  77.56857, 22.52072, 30.01663, 87.35534, 0.5265305, 82.12284, 79.70694,
                    46.7935,  30.30324,  27.84256, 25.48696, 44.50763, 50.45483, 55.34974,  99.55003, 79.26619,
                    62.21792, 98.89601,  21.53087, 16.0212,  61.25396, 4.394201, 3.568028,  51.48888, 46.6206,
                    91.71678, 62.92263,  51.41176, 49.68734, 24.75149, 1.179403, 19.24021,  69.20321, 20.06067,
                    36.95363, 0.3734242, 83.00477, 15.44611, 26.75993, 88.03322, 50.97908,  84.71502, 63.97172,
                    74.17709, 9.149561,  54.11438, 50.77722, 87.13394, 36.12641, 59.81841,  5.925164, 38.76318,
                    32.30363, 15.01997,  81.63381, 37.94462, 97.87479, 58.99917, 60.50563,  63.79966, 67.64502,
                    15.0788,  44.03135,  23.9564,  40.24983, 9.670409, 96.78281, 21.5004,   67.17652, 30.04201,
                    87.4077,  66.22147,  13.16158, 84.50743, 94.49482, 90.39168, 56.97191,  14.546,   19.24635,
                    92.79057, 55.23265,  18.05525}},
            77,
            69,
            op::PhiloxAlignment::TENSORFLOW,
            reference_tests::Tensor{
                {3, 4, 1, 7},
                element::f64,
                std::vector<double>{67.0, 93.0, 76.0, 19.0,  25.0, 65.0, 0.0,  87.0, 78.0, 41.0,  36.0,  20.0,
                                    33.0, 48.0, 36.0, 68.0,  99.0, 65.0, 67.0, 87.0, 19.0, 8.0,   58.0,  12.0,
                                    6.0,  50.0, 42.0, 81.0,  70.0, 56.0, 49.0, 20.0, 1.0,  17.0,  67.0,  29.0,
                                    41.0, 0.0,  64.0, 10.0,  18.0, 77.0, 50.0, 75.0, 55.0, 64.0,  9.0,   49.0,
                                    44.0, 81.0, 27.0, 66.0,  4.0,  47.0, 35.0, 13.0, 88.0, 39.0,  101.0, 56.0,
                                    64.0, 62.0, 61.0, 17.0,  38.0, 28.0, 45.0, 9.0,  93.0, 29.0,  66.0,  26.0,
                                    86.0, 73.0, 17.0, 104.0, 93.0, 83.0, 54.0, 15.0, 16.0, 106.0, 65.0,  15.0}},
            "float64_tensorflow_mixed_knuth_hoermann_4d_tensor")),
    ReferenceRandomPoissonLayerTest::getTestCaseName);
}  // namespace reference_tests