// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/istft.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

namespace {
using ov::Shape;
struct ISTFTParams {
    ISTFTParams(const reference_tests::Tensor& signal,
                const reference_tests::Tensor& window,
                const reference_tests::Tensor& frame_size,
                const reference_tests::Tensor& frame_step,
                const reference_tests::Tensor& length,
                bool center,
                bool normalized,
                const reference_tests::Tensor& expected_tensor,
                std::string name)
        : signal{signal},
          window{window},
          frame_size{frame_size},
          frame_step{frame_step},
          length{length},
          center{center},
          normalized{normalized},
          expected_tensor(expected_tensor),
          test_case_name{std::move(name)} {}

    reference_tests::Tensor signal;
    reference_tests::Tensor window;
    reference_tests::Tensor frame_size;
    reference_tests::Tensor frame_step;
    reference_tests::Tensor length;

    bool center;
    bool normalized;

    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
};

class ReferenceISTFT : public testing::TestWithParam<ISTFTParams>, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        if (shape_size(params.length.shape) == 0) {  // Ignore signal length
            inputData = {params.signal.data, params.window.data, params.frame_size.data, params.frame_step.data};
        } else {
            inputData = {params.signal.data,
                         params.window.data,
                         params.frame_size.data,
                         params.frame_step.data,
                         params.length.data};
        }

        refOutData = {params.expected_tensor.data};
        abs_threshold = 1e-5f;
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ISTFTParams>& obj) {
        std::ostringstream name;
        name << obj.param.test_case_name;
        name << "_signal_input_type_";
        name << obj.param.signal.type;
        name << "_signal_shape_";
        name << obj.param.signal.shape;
        name << "_window_input_type_";
        name << obj.param.window.type;
        name << "_window_shape_";
        name << obj.param.window.shape;
        name << "_frame_size_input_type_";
        name << obj.param.frame_size.type;
        name << "_frame_size_shape_";
        name << obj.param.frame_size.shape;
        name << "_frame_step_input_type_";
        name << obj.param.frame_step.type;
        name << "_frame_step_shape_";
        name << obj.param.frame_step.shape;
        name << "_length_input_type_";
        name << obj.param.frame_step.type;
        name << "_length_shape_";
        name << obj.param.frame_step.shape;
        name << "_center_";
        name << obj.param.center;
        name << "_normalized_";
        name << obj.param.normalized;
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const ISTFTParams& params) {
        const auto in_signal = std::make_shared<ov::op::v0::Parameter>(params.signal.type, params.signal.shape);
        const auto in_window = std::make_shared<ov::op::v0::Parameter>(params.window.type, params.window.shape);
        const auto in_frame_size =
            std::make_shared<ov::op::v0::Parameter>(params.frame_size.type, params.frame_size.shape);
        const auto in_frame_step =
            std::make_shared<ov::op::v0::Parameter>(params.frame_step.type, params.frame_step.shape);
        const auto in_length = std::make_shared<ov::op::v0::Parameter>(params.length.type, params.length.shape);

        std::shared_ptr<ov::op::v16::ISTFT> ISTFT;
        if (shape_size(params.length.shape) == 0) {
            ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
                                                         in_window,
                                                         in_frame_size,
                                                         in_frame_step,
                                                         params.center,
                                                         params.normalized);
            return std::make_shared<ov::Model>(ISTFT->outputs(),
                                               ov::ParameterVector{in_signal, in_window, in_frame_size, in_frame_step});
        } else {
            ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
                                                         in_window,
                                                         in_frame_size,
                                                         in_frame_step,
                                                         in_length,
                                                         params.center,
                                                         params.normalized);
            return std::make_shared<ov::Model>(
                ISTFT->outputs(),
                ov::ParameterVector{in_signal, in_window, in_frame_size, in_frame_step, in_length});
        }
    }
};

template <ov::element::Type_t ET, ov::element::Type_t IT = ov::element::i64>
std::vector<ISTFTParams> generateISTFTParams() {
    using VT = typename ov::element_type_traits<ET>::value_type;
    using INT_T = typename ov::element_type_traits<IT>::value_type;

    const ov::Shape signal_48_shape{48};
    const ov::Shape signal_39_shape{39};
    const ov::Shape signal_1_48_shape{1, 48};
    const ov::Shape signal_2_48_shape{2, 48};
    const ov::Shape signal_256_shape{1, 256};

    const ov::Shape signal_16_shape{16};

    reference_tests::Tensor signal_16(signal_16_shape,
                                      ET,
                                      std::vector<VT>{5.8779e-01,
                                                      -7.7051e-01,
                                                      -6.8455e-01,
                                                      6.8455e-01,
                                                      7.7051e-01,
                                                      -5.8779e-01,
                                                      -8.4433e-01,
                                                      4.8175e-01,
                                                      9.0483e-01,
                                                      -3.6813e-01,
                                                      -9.5106e-01,
                                                      2.4869e-01,
                                                      9.8229e-01,
                                                      -1.2533e-01,
                                                      -9.9803e-01,
                                                      6.8545e-07});

    reference_tests::Tensor signal_48(
        signal_48_shape,
        ET,
        std::vector<VT>{-0.9511, -0.1861, 0.7722,  0.9283,  0.1200,  -0.8129, -0.9014, -0.0534, 0.8500,  0.8704,
                        -0.0134, -0.8833, -0.8356, 0.0801,  0.9126,  0.7971,  -0.1465, -0.9379, -0.7550, 0.2123,
                        0.9590,  0.7095,  -0.2771, -0.9758, -0.6608, 0.3406,  0.9882,  0.6092,  -0.4027, -0.9962,
                        -0.5549, 0.4629,  0.9998,  0.4981,  -0.5211, -0.9989, -0.4390, 0.5769,  0.9936,  0.3780,
                        -0.6302, -0.9838, -0.3154, 0.6806,  0.9696,  0.2513,  -0.7281, -0.9511});

    reference_tests::Tensor signal_48_pad_0_to_55(
        Shape{55},
        ET,
        std::vector<VT>{-0.9511, -0.1861, 0.7722,  0.9283,  0.1200,  -0.8129, -0.9014, -0.0534, 0.8500,  0.8704,
                        -0.0134, -0.8833, -0.8356, 0.0801,  0.9126,  0.7971,  -0.1465, -0.9379, -0.7550, 0.2123,
                        0.9590,  0.7095,  -0.2771, -0.9758, -0.6608, 0.3406,  0.9882,  0.6092,  -0.4027, -0.9962,
                        -0.5549, 0.4629,  0.9998,  0.4981,  -0.5211, -0.9989, -0.4390, 0.5769,  0.9936,  0.3780,
                        -0.6302, -0.9838, -0.3154, 0.6806,  0.9696,  0.2513,  -0.7281, -0.9511, 0.0000,  0.0000,
                        0.0000,  0.0000,  0.0000,  0.0000,  0.0000});

    reference_tests::Tensor signal_39(
        signal_39_shape,
        ET,
        std::vector<VT>{-0.9511, -0.1861, 0.7722,  0.9283,  0.1200,  -0.8129, -0.9014, -0.0534, 0.8500,  0.8704,
                        -0.0134, -0.8833, -0.8356, 0.0801,  0.9126,  0.7971,  -0.1465, -0.9379, -0.7550, 0.2123,
                        0.9590,  0.7095,  -0.2771, -0.9758, -0.6608, 0.3406,  0.9882,  0.6092,  -0.4027, -0.9962,
                        -0.5549, 0.4629,  0.9998,  0.4981,  -0.5211, -0.9989, -0.4390, 0.5769,  0.9936});

    reference_tests::Tensor signal_55(
        Shape{55},
        ET,
        std::vector<VT>{-0.9511, -0.1861, 0.7722,  0.9283,  0.1200,  -0.8129, -0.9014, -0.0534, 0.8500,  0.8704,
                        -0.0134, -0.8833, -0.8356, 0.0801,  0.9126,  0.7971,  -0.1465, -0.9379, -0.7550, 0.2123,
                        0.9590,  0.7095,  -0.2771, -0.9758, -0.6608, 0.3406,  0.9882,  0.6092,  -0.4027, -0.9962,
                        -0.5549, 0.4629,  0.9998,  0.4981,  -0.5211, -0.9989, -0.4390, 0.5769,  0.9936,  0.3780,
                        -0.6302, -0.9838, -0.3154, 0.6806,  0.9696,  0.2513,  -0.7281, -0.9511, -0.7281, 0.2513,
                        0.9696,  0.6806,  -0.3154, -0.9838, -0.6302});

    reference_tests::Tensor signal_60(
        Shape{60},
        ET,
        std::vector<VT>{-0.9511, -0.1861, 0.7722,  0.9283,  0.1200,  -0.8129, -0.9014, -0.0534, 0.8500,  0.8704,
                        -0.0134, -0.8833, -0.8356, 0.0801,  0.9126,  0.7971,  -0.1465, -0.9379, -0.7550, 0.2123,
                        0.9590,  0.7095,  -0.2771, -0.9758, -0.6608, 0.3406,  0.9882,  0.6092,  -0.4027, -0.9962,
                        -0.5549, 0.4629,  0.9998,  0.4981,  -0.5211, -0.9989, -0.4390, 0.5769,  0.9936,  0.3780,
                        -0.6302, -0.9838, -0.3154, 0.6806,  0.9696,  0.2513,  -0.7281, -0.9511, -0.7281, 0.2513,
                        0.9696,  0.6806,  -0.3154, -0.9838, -0.6302, 0.3780,  0.0000,  0.0000,  0.0000,  0.0000});

    reference_tests::Tensor signal_1_48(
        signal_1_48_shape,
        ET,
        std::vector<VT>{-0.9511, -0.1861, 0.7722,  0.9283,  0.1200,  -0.8129, -0.9014, -0.0534, 0.8500,  0.8704,
                        -0.0134, -0.8833, -0.8356, 0.0801,  0.9126,  0.7971,  -0.1465, -0.9379, -0.7550, 0.2123,
                        0.9590,  0.7095,  -0.2771, -0.9758, -0.6608, 0.3406,  0.9882,  0.6092,  -0.4027, -0.9962,
                        -0.5549, 0.4629,  0.9998,  0.4981,  -0.5211, -0.9989, -0.4390, 0.5769,  0.9936,  0.3780,
                        -0.6302, -0.9838, -0.3154, 0.6806,  0.9696,  0.2513,  -0.7281, -0.9511});

    reference_tests::Tensor signal_2_48(
        signal_2_48_shape,
        ET,
        std::vector<VT>{
            -0.9511, -0.1861, 0.7722, 0.9283, 0.1200,  -0.8129, -0.9014, -0.0534, 0.8500, 0.8704, -0.0134, -0.8833,
            -0.8356, 0.0801,  0.9126, 0.7971, -0.1465, -0.9379, -0.7550, 0.2123,  0.9590, 0.7095, -0.2771, -0.9758,
            -0.6608, 0.3406,  0.9882, 0.6092, -0.4027, -0.9962, -0.5549, 0.4629,  0.9998, 0.4981, -0.5211, -0.9989,
            -0.4390, 0.5769,  0.9936, 0.3780, -0.6302, -0.9838, -0.3154, 0.6806,  0.9696, 0.2513, -0.7281, -0.9511,
            -0.9511, -0.1861, 0.7722, 0.9283, 0.1200,  -0.8129, -0.9014, -0.0534, 0.8500, 0.8704, -0.0134, -0.8833,
            -0.8356, 0.0801,  0.9126, 0.7971, -0.1465, -0.9379, -0.7550, 0.2123,  0.9590, 0.7095, -0.2771, -0.9758,
            -0.6608, 0.3406,  0.9882, 0.6092, -0.4027, -0.9962, -0.5549, 0.4629,  0.9998, 0.4981, -0.5211, -0.9989,
            -0.4390, 0.5769,  0.9936, 0.3780, -0.6302, -0.9838, -0.3154, 0.6806,  0.9696, 0.2513, -0.7281, -0.9511});

    reference_tests::Tensor ones_window_16(Shape{16}, ET, std::vector<VT>(16, 1.f));
    reference_tests::Tensor two_window_16(Shape{16}, ET, std::vector<VT>(16, 2.f));

    reference_tests::Tensor hann_window_5(Shape{5}, ET, std::vector<VT>{0., 0.5, 1., 0.5, 0.});
    reference_tests::Tensor hann_window_7(Shape{7}, ET, std::vector<VT>{0., 0.25, 0.75, 1., 0.75, 0.25, 0.});
    reference_tests::Tensor hann_window_8(
        Shape{8},
        ET,
        std::vector<VT>{0., 0.18826, 0.61126, 0.95048, 0.95048, 0.61126, 0.18826, 0.});
    reference_tests::Tensor hann_window_10(
        Shape{10},
        ET,
        std::vector<VT>{0., 0.11698, 0.41318, 0.75, 0.96985, 0.96985, 0.75, 0.41318, 0.11698, 0.});
    reference_tests::Tensor hann_window_16(Shape{16},
                                           ET,
                                           std::vector<VT>{0.,
                                                           0.04323,
                                                           0.16543,
                                                           0.34549,
                                                           0.55226,
                                                           0.75,
                                                           0.90451,
                                                           0.98907,
                                                           0.98907,
                                                           0.90451,
                                                           0.75,
                                                           0.55226,
                                                           0.34549,
                                                           0.16543,
                                                           0.04323,
                                                           0.});

    reference_tests::Tensor hann_window_period_16(Shape{16},
                                                  ET,
                                                  std::vector<VT>{0.0000,
                                                                  0.0381,
                                                                  0.1464,
                                                                  0.3087,
                                                                  0.5000,
                                                                  0.6913,
                                                                  0.8536,
                                                                  0.9619,
                                                                  1.0000,
                                                                  0.9619,
                                                                  0.8536,
                                                                  0.6913,
                                                                  0.5000,
                                                                  0.3087,
                                                                  0.1464,
                                                                  0.0381});

    reference_tests::Tensor frame_size_8(Shape{}, IT, std::vector<INT_T>{8});
    reference_tests::Tensor frame_size_9(Shape{}, IT, std::vector<INT_T>{9});
    reference_tests::Tensor frame_size_11(Shape{}, IT, std::vector<INT_T>{11});
    reference_tests::Tensor frame_size_16(Shape{}, IT, std::vector<INT_T>{16});

    reference_tests::Tensor frame_step_2(Shape{}, IT, std::vector<INT_T>{2});
    reference_tests::Tensor frame_step_3(Shape{}, IT, std::vector<INT_T>{3});
    reference_tests::Tensor frame_step_4(Shape{}, IT, std::vector<INT_T>{4});
    reference_tests::Tensor frame_step_8(Shape{}, IT, std::vector<INT_T>{8});
    reference_tests::Tensor frame_step_12(Shape{}, IT, std::vector<INT_T>{12});
    reference_tests::Tensor frame_step_16(Shape{}, IT, std::vector<INT_T>{16});
    reference_tests::Tensor frame_step_100(Shape{}, IT, std::vector<INT_T>{100});

    reference_tests::Tensor output_stft_9_1_2_transp_win_one(Shape{9, 1, 2},
                                                             ET,
                                                             std::vector<VT>{-0.6693,
                                                                             0.0000,
                                                                             -0.7103,
                                                                             -0.0912,
                                                                             -0.8803,
                                                                             -0.2251,
                                                                             -1.5651,
                                                                             -0.5924,
                                                                             6.7234,
                                                                             3.2667,
                                                                             0.7715,
                                                                             0.4254,
                                                                             0.3599,
                                                                             0.1884,
                                                                             0.2358,
                                                                             0.0796,
                                                                             0.2042,
                                                                             0.0000});

    reference_tests::Tensor output_stft_9_1_2_transp_win_two(Shape{9, 1, 2},
                                                             ET,
                                                             std::vector<VT>{-1.3386,
                                                                             0.0000,
                                                                             -1.4207,
                                                                             -0.1823,
                                                                             -1.7606,
                                                                             -0.4502,
                                                                             -3.1302,
                                                                             -1.1848,
                                                                             13.4467,
                                                                             6.5335,
                                                                             1.5429,
                                                                             0.8508,
                                                                             0.7199,
                                                                             0.3768,
                                                                             0.4716,
                                                                             0.1591,
                                                                             0.4084,
                                                                             0.0000});

    reference_tests::Tensor output_stft_9_3_2_transp_win_two(
        Shape{9, 3, 2},
        ET,
        std::vector<VT>{1.3873,   0.0000,  -2.8503, 0.0000,  -0.4391, 0.0000,   1.7637,  -0.6945, -3.1429,
                        -0.9896,  -0.7182, 1.0237,  4.2213,  -2.5114, -5.0535,  -3.5783, -2.5402, 3.7017,
                        -12.4337, 7.5925,  7.8944,  10.8180, 9.8076,  -11.1912, -3.1735, 1.6741,  0.6953,
                        2.3853,   2.9422,  -2.4676, -2.1233, 0.8610,  -0.1211,  1.2268,  2.1636,  -1.2691,
                        -1.7631,  0.4790,  -0.4011, 0.6825,  1.8965,  -0.7060,  -1.6151, 0.2192,  -0.5161,
                        0.3123,   1.7869,  -0.3231, -1.5735, 0.0000,  -0.5485,  0.0000,  1.7559,  0.0000});

    reference_tests::Tensor output_stft_2_9_3_2_transp_win_two(
        Shape{2, 9, 3, 2},
        ET,
        std::vector<VT>{1.3873,  0.0000,  -2.8503,  0.0000,   -0.4391, 0.0000,  1.7637,  -0.6945,  -3.1429,  -0.9896,
                        -0.7182, 1.0237,  4.2213,   -2.5114,  -5.0535, -3.5783, -2.5402, 3.7017,   -12.4337, 7.5925,
                        7.8944,  10.8180, 9.8076,   -11.1912, -3.1735, 1.6741,  0.6953,  2.3853,   2.9422,   -2.4676,
                        -2.1233, 0.8610,  -0.1211,  1.2268,   2.1636,  -1.2691, -1.7631, 0.4790,   -0.4011,  0.6825,
                        1.8965,  -0.7060, -1.6151,  0.2192,   -0.5161, 0.3123,  1.7869,  -0.3231,  -1.5735,  0.0000,
                        -0.5485, 0.0000,  1.7559,   0.0000,   1.3873,  0.0000,  -2.8503, 0.0000,   -0.4391,  0.0000,
                        1.7637,  -0.6945, -3.1429,  -0.9896,  -0.7182, 1.0237,  4.2213,  -2.5114,  -5.0535,  -3.5783,
                        -2.5402, 3.7017,  -12.4337, 7.5925,   7.8944,  10.8180, 9.8076,  -11.1912, -3.1735,  1.6741,
                        0.6953,  2.3853,  2.9422,   -2.4676,  -2.1233, 0.8610,  -0.1211, 1.2268,   2.1636,   -1.2691,
                        -1.7631, 0.4790,  -0.4011,  0.6825,   1.8965,  -0.7060, -1.6151, 0.2192,   -0.5161,  0.3123,
                        1.7869,  -0.3231, -1.5735,  0.0000,   -0.5485, 0.0000,  1.7559,  0.0000});

    reference_tests::Tensor output_stft_9_4_2_transp_win_two_center(
        Shape{9, 4, 2},
        ET,
        std::vector<VT>{
            -7.3526e-01, 0.0000e+00, 1.1330e+00, 0.0000e+00,  2.5475e+00,  0.0000e+00,  -4.1695e+00, 0.0000e+00,
            -3.3068e+00, 2.3842e-07, 1.0681e+00, 1.3042e+00,  2.9902e+00,  -2.6441e-02, -2.2545e+00, -9.3384e-01,
            -1.6860e+00, 0.0000e+00, 6.4448e-01, 4.7161e+00,  5.8809e+00,  -9.5572e-02, -6.7604e+00, -6.7604e+00,
            1.4974e+01,  0.0000e+00, 3.5154e+00, -1.4258e+01, -1.3709e+01, 2.8896e-01,  4.2285e+00,  1.0209e+01,
            7.9468e-01,  0.0000e+00, 1.9192e+00, -3.1437e+00, -2.8170e+00, 6.3723e-02,  0.0000e+00,  4.5065e+00,
            1.6981e+00,  0.0000e+00, 1.7382e+00, -1.6169e+00, -1.5818e+00, 3.2759e-02,  -4.7954e-01, 1.1577e+00,
            3.2156e-01,  0.0000e+00, 1.6761e+00, -8.9949e-01, -1.1581e+00, 1.8237e-02,  -1.2894e+00, 1.2894e+00,
            1.0437e+00,  2.3842e-07, 1.6506e+00, -4.1167e-01, -9.8408e-01, 8.3490e-03,  -7.1159e-01, 2.9475e-01,
            2.5795e-01,  0.0000e+00, 1.6434e+00, 0.0000e+00,  -9.3507e-01, 0.0000e+00,  -1.4628e+00, 0.0000e+00});

    reference_tests::Tensor output_stft_2_9_4_2_transp_win_two_center(
        Shape{2, 9, 4, 2},
        ET,
        std::vector<VT>{
            -7.3526e-01, 0.0000e+00, 1.1330e+00, 0.0000e+00,  2.5475e+00,  0.0000e+00,  -4.1695e+00, 0.0000e+00,
            -3.3068e+00, 2.3842e-07, 1.0681e+00, 1.3042e+00,  2.9902e+00,  -2.6441e-02, -2.2545e+00, -9.3384e-01,
            -1.6860e+00, 0.0000e+00, 6.4448e-01, 4.7161e+00,  5.8809e+00,  -9.5572e-02, -6.7604e+00, -6.7604e+00,
            1.4974e+01,  0.0000e+00, 3.5154e+00, -1.4258e+01, -1.3709e+01, 2.8896e-01,  4.2285e+00,  1.0209e+01,
            7.9468e-01,  0.0000e+00, 1.9192e+00, -3.1437e+00, -2.8170e+00, 6.3723e-02,  0.0000e+00,  4.5065e+00,
            1.6981e+00,  0.0000e+00, 1.7382e+00, -1.6169e+00, -1.5818e+00, 3.2759e-02,  -4.7954e-01, 1.1577e+00,
            3.2156e-01,  0.0000e+00, 1.6761e+00, -8.9949e-01, -1.1581e+00, 1.8237e-02,  -1.2894e+00, 1.2894e+00,
            1.0437e+00,  2.3842e-07, 1.6506e+00, -4.1167e-01, -9.8408e-01, 8.3490e-03,  -7.1159e-01, 2.9475e-01,
            2.5795e-01,  0.0000e+00, 1.6434e+00, 0.0000e+00,  -9.3507e-01, 0.0000e+00,  -1.4628e+00, 0.0000e+00,
            -7.3526e-01, 0.0000e+00, 1.1330e+00, 0.0000e+00,  2.5475e+00,  0.0000e+00,  -4.1695e+00, 0.0000e+00,
            -3.3068e+00, 2.3842e-07, 1.0681e+00, 1.3042e+00,  2.9902e+00,  -2.6441e-02, -2.2545e+00, -9.3384e-01,
            -1.6860e+00, 0.0000e+00, 6.4448e-01, 4.7161e+00,  5.8809e+00,  -9.5572e-02, -6.7604e+00, -6.7604e+00,
            1.4974e+01,  0.0000e+00, 3.5154e+00, -1.4258e+01, -1.3709e+01, 2.8896e-01,  4.2285e+00,  1.0209e+01,
            7.9468e-01,  0.0000e+00, 1.9192e+00, -3.1437e+00, -2.8170e+00, 6.3723e-02,  0.0000e+00,  4.5065e+00,
            1.6981e+00,  0.0000e+00, 1.7382e+00, -1.6169e+00, -1.5818e+00, 3.2759e-02,  -4.7954e-01, 1.1577e+00,
            3.2156e-01,  0.0000e+00, 1.6761e+00, -8.9949e-01, -1.1581e+00, 1.8237e-02,  -1.2894e+00, 1.2894e+00,
            1.0437e+00,  2.3842e-07, 1.6506e+00, -4.1167e-01, -9.8408e-01, 8.3490e-03,  -7.1159e-01, 2.9475e-01,
            2.5795e-01,  0.0000e+00, 1.6434e+00, 0.0000e+00,  -9.3507e-01, 0.0000e+00,  -1.4628e+00, 0.0000e+00});

    reference_tests::Tensor output_stft_9_4_2_transp_win_two_center_norm(
        Shape{9, 4, 2},
        ET,
        std::vector<VT>{
            -1.8382e-01, 0.0000e+00, 2.8325e-01, 0.0000e+00,  6.3686e-01,  0.0000e+00,  -1.0424e+00, 0.0000e+00,
            -8.2669e-01, 5.9605e-08, 2.6703e-01, 3.2606e-01,  7.4754e-01,  -6.6102e-03, -5.6362e-01, -2.3346e-01,
            -4.2149e-01, 0.0000e+00, 1.6112e-01, 1.1790e+00,  1.4702e+00,  -2.3893e-02, -1.6901e+00, -1.6901e+00,
            3.7434e+00,  0.0000e+00, 8.7886e-01, -3.5645e+00, -3.4273e+00, 7.2241e-02,  1.0571e+00,  2.5522e+00,
            1.9867e-01,  0.0000e+00, 4.7980e-01, -7.8593e-01, -7.0426e-01, 1.5931e-02,  0.0000e+00,  1.1266e+00,
            4.2452e-01,  0.0000e+00, 4.3454e-01, -4.0423e-01, -3.9546e-01, 8.1899e-03,  -1.1988e-01, 2.8943e-01,
            8.0389e-02,  0.0000e+00, 4.1901e-01, -2.2487e-01, -2.8953e-01, 4.5592e-03,  -3.2235e-01, 3.2235e-01,
            2.6093e-01,  5.9605e-08, 4.1264e-01, -1.0292e-01, -2.4602e-01, 2.0872e-03,  -1.7790e-01, 7.3688e-02,
            6.4488e-02,  0.0000e+00, 4.1084e-01, 0.0000e+00,  -2.3377e-01, 0.0000e+00,  -3.6569e-01, 0.0000e+00});

    reference_tests::Tensor output_stft_9_5_2_transp_win_hann_period_center_norm(
        Shape{9, 5, 2},
        ET,
        std::vector<VT>{1.6072e-01,  0.0000e+00,  2.3125e-02,  0.0000e+00,  1.8288e-02,  0.0000e+00,  1.2149e-02,
                        0.0000e+00,  -1.1969e-01, 0.0000e+00,  -1.3101e-01, -2.9802e-08, 6.3939e-02,  3.6574e-02,
                        5.0562e-02,  4.9973e-02,  3.3594e-02,  5.9822e-02,  2.0066e-01,  1.5290e-01,  -4.6996e-01,
                        1.0431e-07,  -5.8716e-01, -3.8846e-01, -4.6433e-01, -5.3078e-01, -3.0849e-01, -6.3539e-01,
                        -4.8422e-01, -7.1236e-01, 9.6370e-01,  1.1176e-08,  7.9614e-01,  5.2212e-01,  6.2959e-01,
                        7.1341e-01,  4.1829e-01,  8.5400e-01,  4.7555e-01,  7.0847e-01,  -4.7132e-01, -6.7055e-08,
                        -2.5222e-01, -1.6637e-01, -1.9946e-01, -2.2732e-01, -1.3252e-01, -2.7212e-01, -1.1716e-01,
                        -7.3543e-02, 7.1247e-02,  1.1176e-08,  -2.1195e-02, -1.4046e-02, -1.6761e-02, -1.9191e-02,
                        -1.1135e-02, -2.2974e-02, 1.0323e-02,  -1.0876e-01, -6.5584e-02, -1.4901e-08, -6.5221e-03,
                        -3.9840e-03, -5.1576e-03, -5.4441e-03, -3.4269e-03, -6.5171e-03, -4.3364e-02, 3.5198e-02,
                        4.7124e-02,  -2.9802e-08, -3.2645e-03, -1.3218e-03, -2.5820e-03, -1.8056e-03, -1.7148e-03,
                        -2.1611e-03, 4.1531e-02,  -2.1872e-02, -4.9111e-02, 0.0000e+00,  -2.5608e-03, 0.0000e+00,
                        -2.0245e-03, 0.0000e+00,  -1.3463e-03, 0.0000e+00,  -4.6949e-02, 0.0000e+00});

    reference_tests::Tensor output_stft_9_5_2_transp_win_hann_period_center(
        Shape{9, 5, 2},
        ET,
        std::vector<VT>{6.4288e-01,  0.0000e+00,  9.2500e-02,  0.0000e+00,  7.3152e-02,  0.0000e+00,  4.8597e-02,
                        0.0000e+00,  -4.7876e-01, 0.0000e+00,  -5.2404e-01, -1.1921e-07, 2.5575e-01,  1.4629e-01,
                        2.0225e-01,  1.9989e-01,  1.3437e-01,  2.3929e-01,  8.0262e-01,  6.1159e-01,  -1.8798e+00,
                        4.1723e-07,  -2.3486e+00, -1.5538e+00, -1.8573e+00, -2.1231e+00, -1.2340e+00, -2.5415e+00,
                        -1.9369e+00, -2.8494e+00, 3.8548e+00,  4.4703e-08,  3.1846e+00,  2.0885e+00,  2.5184e+00,
                        2.8536e+00,  1.6732e+00,  3.4160e+00,  1.9022e+00,  2.8339e+00,  -1.8853e+00, -2.6822e-07,
                        -1.0089e+00, -6.6547e-01, -7.9783e-01, -9.0929e-01, -5.3007e-01, -1.0885e+00, -4.6863e-01,
                        -2.9417e-01, 2.8499e-01,  4.4703e-08,  -8.4780e-02, -5.6183e-02, -6.7043e-02, -7.6764e-02,
                        -4.4541e-02, -9.1897e-02, 4.1290e-02,  -4.3506e-01, -2.6234e-01, -5.9605e-08, -2.6088e-02,
                        -1.5936e-02, -2.0630e-02, -2.1776e-02, -1.3708e-02, -2.6068e-02, -1.7346e-01, 1.4079e-01,
                        1.8850e-01,  -1.1921e-07, -1.3058e-02, -5.2870e-03, -1.0328e-02, -7.2225e-03, -6.8594e-03,
                        -8.6446e-03, 1.6612e-01,  -8.7487e-02, -1.9645e-01, 0.0000e+00,  -1.0243e-02, 0.0000e+00,
                        -8.0982e-03, 0.0000e+00,  -5.3853e-03, 0.0000e+00,  -1.8780e-01, 0.0000e+00});

    reference_tests::Tensor output_stft_9_5_2_transp_win_hann_center(
        Shape{9, 5, 2},
        ET,
        std::vector<VT>{6.9553e-01,  0.0000e+00,  1.6603e-01,  0.0000e+00,  1.5808e-01,  0.0000e+00,  1.3889e-01,
                        0.0000e+00,  -4.2351e-01, 0.0000e+00,  -3.9394e-01, -5.8741e-02, 4.4552e-01,  1.0763e-01,
                        4.0360e-01,  2.0588e-01,  3.3302e-01,  2.8950e-01,  1.0469e+00,  6.2184e-01,  -1.8603e+00,
                        4.5327e-01,  -2.4868e+00, -1.1705e+00, -2.0868e+00, -1.7925e+00, -1.5385e+00, -2.2871e+00,
                        -2.1314e+00, -2.5705e+00, 3.6116e+00,  4.9244e-02,  2.8938e+00,  2.1352e+00,  2.2286e+00,
                        2.8211e+00,  1.4049e+00,  3.3065e+00,  1.5360e+00,  2.8408e+00,  -1.9225e+00, -3.4335e-01,
                        -8.9832e-01, -9.8693e-01, -6.0476e-01, -1.1902e+00, -2.6823e-01, -1.3088e+00, -2.8283e-01,
                        -5.4008e-01, 3.4255e-01,  1.2559e-02,  -3.2922e-02, -5.4718e-02, -1.7115e-02, -6.1602e-02,
                        -9.1553e-05, -6.4114e-02, 1.1357e-01,  -4.4302e-01, -2.3653e-01, -5.6663e-03, -3.8263e-03,
                        -1.0372e-02, -1.3686e-03, -1.0818e-02, 1.1847e-03,  -1.0495e-02, -1.6184e-01, 1.5109e-01,
                        2.0243e-01,  3.8466e-03,  -4.6578e-04, -2.0989e-03, -7.1570e-04, -1.9433e-03, -9.1043e-04,
                        -1.6500e-03, 1.8387e-01,  -8.4278e-02, -1.8225e-01, 0.0000e+00,  -1.4386e-04, 0.0000e+00,
                        -9.4226e-04, 0.0000e+00,  -1.6809e-03, 0.0000e+00,  -1.8522e-01, 0.0000e+00});

    reference_tests::Tensor output_stft_9_5_2_transp_win_hann_center_norm(
        Shape{9, 5, 2},
        ET,
        std::vector<VT>{1.7388e-01,  0.0000e+00,  4.1509e-02,  0.0000e+00,  3.9521e-02,  0.0000e+00,  3.4723e-02,
                        0.0000e+00,  -1.0588e-01, 0.0000e+00,  -9.8485e-02, -1.4685e-02, 1.1138e-01,  2.6907e-02,
                        1.0090e-01,  5.1469e-02,  8.3255e-02,  7.2374e-02,  2.6173e-01,  1.5546e-01,  -4.6508e-01,
                        1.1332e-01,  -6.2169e-01, -2.9264e-01, -5.2170e-01, -4.4813e-01, -3.8463e-01, -5.7178e-01,
                        -5.3284e-01, -6.4262e-01, 9.0291e-01,  1.2311e-02,  7.2346e-01,  5.3381e-01,  5.5715e-01,
                        7.0528e-01,  3.5124e-01,  8.2664e-01,  3.8400e-01,  7.1019e-01,  -4.8062e-01, -8.5839e-02,
                        -2.2458e-01, -2.4673e-01, -1.5119e-01, -2.9754e-01, -6.7057e-02, -3.2721e-01, -7.0707e-02,
                        -1.3502e-01, 8.5637e-02,  3.1397e-03,  -8.2305e-03, -1.3679e-02, -4.2788e-03, -1.5400e-02,
                        -2.2888e-05, -1.6028e-02, 2.8393e-02,  -1.1075e-01, -5.9133e-02, -1.4166e-03, -9.5657e-04,
                        -2.5929e-03, -3.4216e-04, -2.7046e-03, 2.9618e-04,  -2.6238e-03, -4.0460e-02, 3.7772e-02,
                        5.0607e-02,  9.6166e-04,  -1.1645e-04, -5.2471e-04, -1.7893e-04, -4.8584e-04, -2.2761e-04,
                        -4.1250e-04, 4.5969e-02,  -2.1069e-02, -4.5562e-02, 0.0000e+00,  -3.5964e-05, 0.0000e+00,
                        -2.3557e-04, 0.0000e+00,  -4.2021e-04, 0.0000e+00,  -4.6305e-02, 0.0000e+00});

    reference_tests::Tensor auto_length(Shape{0}, IT, std::vector<INT_T>{});
    reference_tests::Tensor length_16(Shape{}, IT, std::vector<INT_T>{16});
    reference_tests::Tensor length_39(Shape{}, IT, std::vector<INT_T>{39});
    reference_tests::Tensor length_48(Shape{}, IT, std::vector<INT_T>{48});
    reference_tests::Tensor length_55(Shape{}, IT, std::vector<INT_T>{55});
    reference_tests::Tensor length_60(Shape{}, IT, std::vector<INT_T>{60});

    std::vector<ISTFTParams> params;
    params.emplace_back(output_stft_9_1_2_transp_win_one,
                        ones_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        false,
                        false,
                        signal_16,
                        "basic_1D_transp_ones_win_step_16");
    params.emplace_back(output_stft_9_1_2_transp_win_one,
                        ones_window_16,
                        frame_size_16,
                        frame_step_4,
                        auto_length,
                        false,
                        false,
                        signal_16,
                        "basic_1D_transp_ones_win_step_4");
    params.emplace_back(output_stft_9_1_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        false,
                        false,
                        signal_16,
                        "basic_1D_transp_two_win_step_16");
    params.emplace_back(output_stft_9_1_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_4,
                        auto_length,
                        false,
                        false,
                        signal_16,
                        "basic_1D_transp_two_win_step_4");
    params.emplace_back(output_stft_9_3_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        false,
                        false,
                        signal_48,
                        "basic_1D_transp_two_win_step_16");

    params.emplace_back(output_stft_9_3_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_48,
                        false,
                        false,
                        signal_48,
                        "basic_1D_transp_two_win_step_16_length_48");

    params.emplace_back(output_stft_9_3_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_55,
                        false,
                        false,
                        signal_48_pad_0_to_55,
                        "basic_1D_transp_two_win_step_16_length_55");

    params.emplace_back(output_stft_9_3_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_39,
                        false,
                        false,
                        signal_39,
                        "basic_1D_transp_two_win_step_16_length_39");

    params.emplace_back(output_stft_2_9_3_2_transp_win_two,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        false,
                        false,
                        signal_2_48,
                        "basic_2D_transp_two_win_step_16");
    params.emplace_back(output_stft_9_4_2_transp_win_two_center,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        true,
                        false,
                        signal_48,
                        "basic_1D_transp_two_win_step_16_center");
    params.emplace_back(output_stft_9_4_2_transp_win_two_center_norm,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        true,
                        true,
                        signal_48,
                        "basic_1D_transp_two_win_step_16_center_norm");
    params.emplace_back(output_stft_2_9_4_2_transp_win_two_center,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        true,
                        false,
                        signal_2_48,
                        "basic_2D_transp_two_win_step_16_center");
    params.emplace_back(output_stft_2_9_4_2_transp_win_two_center,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_48,
                        true,
                        false,
                        signal_2_48,
                        "basic_2D_transp_two_win_step_16_center_length_48");
    params.emplace_back(output_stft_9_4_2_transp_win_two_center_norm,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_39,
                        true,
                        true,
                        signal_39,
                        "basic_1D_transp_two_win_step_16_center_norm_length_39");

    params.emplace_back(output_stft_9_4_2_transp_win_two_center_norm,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_55,
                        true,
                        true,
                        signal_55,
                        "basic_1D_transp_two_win_step_16_center_norm_length_55");

    params.emplace_back(output_stft_9_4_2_transp_win_two_center_norm,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        length_60,
                        true,
                        true,
                        signal_60,
                        "basic_1D_transp_two_win_step_16_center_norm_length_60");

    params.emplace_back(output_stft_9_5_2_transp_win_hann_center_norm,
                        hann_window_16,
                        frame_size_16,
                        frame_step_12,
                        auto_length,
                        true,
                        true,
                        signal_48,
                        "basic_1D_transp_hann_win_step_16");

    params.emplace_back(output_stft_9_5_2_transp_win_hann_center,
                        hann_window_16,
                        frame_size_16,
                        frame_step_12,
                        auto_length,
                        true,
                        false,
                        signal_48,
                        "basic_1D_transp_hann_win_step_16");

    params.emplace_back(output_stft_9_5_2_transp_win_hann_period_center_norm,
                        hann_window_period_16,
                        frame_size_16,
                        frame_step_12,
                        auto_length,
                        true,
                        true,
                        signal_48,
                        "basic_1D_transp_hann_period_win_step_16");

    params.emplace_back(output_stft_9_5_2_transp_win_hann_period_center,
                        hann_window_period_16,
                        frame_size_16,
                        frame_step_12,
                        auto_length,
                        true,
                        false,
                        signal_48,
                        "basic_1D_transp_hann_period_win_step_16");

    return params;
}

std::vector<ISTFTParams> generateISTFTParams() {
    std::vector<std::vector<ISTFTParams>> combo_params{generateISTFTParams<ov::element::f32>()};
    std::vector<ISTFTParams> test_params;
    for (auto& params : combo_params)
        std::move(params.begin(), params.end(), std::back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceISTFT, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceISTFT,
                         ::testing::ValuesIn(generateISTFTParams()),
                         ReferenceISTFT::getTestCaseName);
