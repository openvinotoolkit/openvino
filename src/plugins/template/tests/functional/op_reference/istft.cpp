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
        inputData = {params.signal.data,
                     params.window.data,
                     params.frame_size.data,
                     params.frame_step.data,
                     params.length.data};
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
        name << "_lentgh_input_type_";
        name << obj.param.frame_step.type;
        name << "_lebgth_shape_";
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

        const auto ISTFT = std::make_shared<ov::op::v16::ISTFT>(in_signal,
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
};

template <ov::element::Type_t ET, ov::element::Type_t IT = ov::element::i64>
std::vector<ISTFTParams> generateISTFTParams() {
    using VT = typename ov::element_type_traits<ET>::value_type;
    using INT_T = typename ov::element_type_traits<IT>::value_type;

    const ov::Shape signal_48_shape{48};
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

    reference_tests::Tensor frame_size_8(Shape{}, IT, std::vector<INT_T>{8});
    reference_tests::Tensor frame_size_9(Shape{}, IT, std::vector<INT_T>{9});
    reference_tests::Tensor frame_size_11(Shape{}, IT, std::vector<INT_T>{11});
    reference_tests::Tensor frame_size_16(Shape{}, IT, std::vector<INT_T>{16});

    reference_tests::Tensor frame_step_2(Shape{}, IT, std::vector<INT_T>{2});
    reference_tests::Tensor frame_step_3(Shape{}, IT, std::vector<INT_T>{3});
    reference_tests::Tensor frame_step_4(Shape{}, IT, std::vector<INT_T>{4});
    reference_tests::Tensor frame_step_8(Shape{}, IT, std::vector<INT_T>{8});
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

    reference_tests::Tensor auto_length(Shape{}, IT, std::vector<INT_T>{-1});
    reference_tests::Tensor length_16(Shape{}, IT, std::vector<INT_T>{16});
    reference_tests::Tensor length_48(Shape{}, IT, std::vector<INT_T>{48});

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
    params.emplace_back(output_stft_9_4_2_transp_win_two_center,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        true,
                        false,
                        signal_48,
                        "basic_1D_transp_two_win_step_16_center");
    params.emplace_back(output_stft_2_9_4_2_transp_win_two_center,
                        two_window_16,
                        frame_size_16,
                        frame_step_16,
                        auto_length,
                        true,
                        false,
                        signal_2_48,
                        "basic_2D_transp_two_win_step_16_center");

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
