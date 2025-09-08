// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocessing/resize_tests.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace {
const Shape target_shape = Shape{1, 1, 8, 8};
std::shared_ptr<ov::Model> build_test_model() {
    const auto input = std::make_shared<op::v0::Parameter>(element::f32, target_shape);
    const auto zero = op::v0::Constant::create(element::f32, Shape{}, {0.0f});
    const auto op = std::make_shared<op::v1::Add>(input, zero);
    const auto res = std::make_shared<op::v0::Result>(op);
    return std::make_shared<ov::Model>(res, ParameterVector{input});
}

ov::Tensor get_input_tensor() {
    ov::Tensor input_tensor(element::f32, Shape{1, 1, 4, 4});
    const auto input_values = std::vector<float>{9.289,
                                                 62.019,
                                                 -35.968,
                                                 44.174,
                                                 97.981,
                                                 -71.053,
                                                 72.736,
                                                 19.516,
                                                 -1.725,
                                                 62.813,
                                                 60.578,
                                                 -65.033,
                                                 -4.954,
                                                 8.976,
                                                 67.107,
                                                 -89.213};
    auto* dst = input_tensor.data<float>();
    std::copy(input_values.begin(), input_values.end(), dst);
    return input_tensor;
}
}  // namespace

namespace preprocess {
std::ostream& operator<<(std::ostream& s, const ResizeAlgorithm& algo) {
    static std::map<preprocess::ResizeAlgorithm, std::string> enum_names = {
        {preprocess::ResizeAlgorithm::RESIZE_LINEAR, "RESIZE_LINEAR"},
        {preprocess::ResizeAlgorithm::RESIZE_CUBIC, "RESIZE_CUBIC"},
        {preprocess::ResizeAlgorithm::RESIZE_NEAREST, "RESIZE_NEAREST"},
        {preprocess::ResizeAlgorithm::RESIZE_BILINEAR_PILLOW, "RESIZE_BILINEAR_PILLOW"},
        {preprocess::ResizeAlgorithm::RESIZE_BICUBIC_PILLOW, "RESIZE_BICUBIC_PILLOW"}};

    return s << enum_names[algo];
}

std::string PreprocessingResizeTests::getTestCaseName(const testing::TestParamInfo<ResizeTestsParams>& obj) {
    std::ostringstream result;
    result << "device=" << std::get<0>(obj.param);
    return result.str();
}

void PreprocessingResizeTests::SetUp() {
    const auto& test_params = this->GetParam();
    this->targetDevice = std::get<0>(test_params);

    this->function = build_test_model();
}

void PreprocessingResizeTests::run() {
    compile_model();
    // inference, output tensors retrieval and tensors comparison:
    validate();
}

void PreprocessingResizeTests::run_with_algorithm(const ResizeAlgorithm algo,
                                                  const std::vector<float>& expected_output) {
    PrePostProcessor ppp(this->function);
    ppp.input().tensor().set_shape({1, 1, -1, -1}).set_layout("NCHW");
    ppp.input(0).preprocess().resize(algo);
    ppp.build();

    this->inputs.insert({this->function->get_parameters().at(0), get_input_tensor()});

    ov::Tensor out_tensor(element::f32, target_shape);
    auto* dst = out_tensor.data<float>();
    std::copy(expected_output.begin(), expected_output.end(), dst);
    expected_output_tensor = out_tensor;

    run();
}

ov::TensorVector PreprocessingResizeTests::calculate_refs() {
    return {expected_output_tensor};
}

TEST_P(PreprocessingResizeTests, Linear) {
    const auto expected_output = std::vector<float>{
        9.289,   22.4715,  48.8365,  37.5223, -11.4712, -15.9325, 24.1385,  44.174,   31.462,   30.7843,  29.4288,
        19.3653, 0.593752, 2.90838,  26.3091, 38.0095,  75.808,   47.4098,  -9.38675, -16.9487, 24.7238,  40.5901,
        30.6504, 25.6805,  73.0545,  45.3943, -9.92625, -10.7658, 42.8757,  51.8671,  16.2082,  -1.62125, 23.2015,
        24.7378, 27.8102,  37.9142,  55.0498, 36.7392,  -17.0174, -43.8957, -2.53225, 10.4392,  36.3822,  52.5679,
        58.9961, 28.8882,  -37.7559, -71.078, -4.14675, 2.49875,  15.7897,  33.1951,  54.7149,  28.3141,  -46.0073,
        -83.168, -4.954,   -1.4715,  5.4935,  23.5088,  52.5743,  28.027,   -50.133,  -89.213};
    run_with_algorithm(ResizeAlgorithm::RESIZE_LINEAR, expected_output);
}

TEST_P(PreprocessingResizeTests, Nearest) {
    const auto expected_output = std::vector<float>{
        9.289,   9.289,   62.019,  62.019,  -35.968, -35.968, 44.174,  44.174,  9.289,   9.289,  62.019,
        62.019,  -35.968, -35.968, 44.174,  44.174,  97.981,  97.981,  -71.053, -71.053, 72.736, 72.736,
        19.516,  19.516,  97.981,  97.981,  -71.053, -71.053, 72.736,  72.736,  19.516,  19.516, -1.725,
        -1.725,  62.813,  62.813,  60.578,  60.578,  -65.033, -65.033, -1.725,  -1.725,  62.813, 62.813,
        60.578,  60.578,  -65.033, -65.033, -4.954,  -4.954,  8.976,   8.976,   67.107,  67.107, -89.213,
        -89.213, -4.954,  -4.954,  8.976,   8.976,   67.107,  67.107,  -89.213, -89.213};
    run_with_algorithm(ResizeAlgorithm::RESIZE_NEAREST, expected_output);
}

TEST_P(PreprocessingResizeTests, Cubic) {
    const auto expected_output = std::vector<float>{
        -8.09343, 21.5218,  71.8322,  52.7927,  -26.7153, -39.113,  21.0894,  56.7106,  33.4924,  32.9301,  32.4765,
        16.0885,  -8.02117, -3.03992, 28.333,   47.094,   103.529,  52.2371,  -33.5401, -45.9531, 22.3128,  56.92,
        41.1729,  32.3988,  99.3848,  48.9419,  -35.8044, -34.1096, 54.7092,  76.6427,  22.0393,  -10.1027, 23.3571,
        25.2392,  27.9298,  45.8128,  70.9863,  44.8728,  -17.9632, -55.8651, -21.1681, 4.77217,  47.5464,  76.9619,
        79.1909,  28.5925,  -47.8029, -94.2156, -11.2722, -1.49591, 14.1184,  43.6733,  73.4948,  34.9309,  -50.9082,
        -102.729, -6.20079, -5.61533, -5.45134, 24.4264,  70.3134,  38.4481,  -53.3457, -108.592};
    run_with_algorithm(ResizeAlgorithm::RESIZE_CUBIC, expected_output);
}

TEST_P(PreprocessingResizeTests, BilinearPillow) {
    const auto expected_output = std::vector<float>{
        9.289,   22.4715,  48.8365,  37.5223, -11.4712, -15.9325, 24.1385,  44.174,   31.462,   30.7843,  29.4288,
        19.3653, 0.593752, 2.90838,  26.3091, 38.0095,  75.808,   47.4098,  -9.38675, -16.9487, 24.7238,  40.5901,
        30.6504, 25.6805,  73.0545,  45.3943, -9.92625, -10.7658, 42.8757,  51.8671,  16.2082,  -1.62125, 23.2015,
        24.7378, 27.8102,  37.9142,  55.0498, 36.7392,  -17.0174, -43.8957, -2.53225, 10.4392,  36.3822,  52.5679,
        58.9961, 28.8882,  -37.7559, -71.078, -4.14675, 2.49875,  15.7897,  33.1951,  54.7149,  28.3141,  -46.0073,
        -83.168, -4.954,   -1.4715,  5.4935,  23.5088,  52.5743,  28.027,   -50.133,  -89.213};
    run_with_algorithm(ResizeAlgorithm::RESIZE_BILINEAR_PILLOW, expected_output);
}

TEST_P(PreprocessingResizeTests, BicubicPillow) {
    const auto expected_output = std::vector<float>{
        -4.91595, 17.7968,  65.9534,  52.453,   -26.0905, -33.4108, 26.2942,  54.4593,  27.8166, 30.4246,  35.9691,
        22.9657,  -8.97529, -5.94229, 29.5679,  46.3238,  97.1959,  57.1927,  -27.5777, -39.541, 27.2632,  52.2528,
        36.5331,  29.1313,  96.0397,  55.3901,  -30.7618, -33.3757, 53.6059,  69.715,   18.5491, -5.58227, 19.4699,
        23.1348,  30.8848,  45.8565,  67.5048,  42.3524,  -24.8737, -56.6079, -15.4118, 3.85346, 44.6572,  68.4128,
        72.2512,  30.8504,  -50.0979, -88.3165, -9.05241, -2.03953, 12.7955,  36.5346,  68.1348, 34.9227,  -56.6621,
        -99.8941, -6.07402, -4.83229, -2.22921, 21.5082,  66.1969,  36.8377,  -59.7761, -105.379};
    run_with_algorithm(ResizeAlgorithm::RESIZE_BICUBIC_PILLOW, expected_output);
}

}  // namespace preprocess
}  // namespace ov
