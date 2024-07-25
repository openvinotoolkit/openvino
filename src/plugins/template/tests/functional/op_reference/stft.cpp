// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/stft.hpp"

#include "base_reference_test.hpp"
#include "gtest/gtest.h"
#include "openvino/op/parameter.hpp"

namespace {
using ov::Shape;
struct STFTParams {
    STFTParams(const reference_tests::Tensor& signal,
               const reference_tests::Tensor& window,
               const reference_tests::Tensor& frame_size,
               const reference_tests::Tensor& frame_step,
               bool transpose_frames,
               const reference_tests::Tensor& expected_tensor,
               std::string name)
        : signal{signal},
          window{window},
          frame_size{frame_size},
          frame_step{frame_step},
          transpose_frames(transpose_frames),
          expected_tensor(expected_tensor),
          test_case_name{std::move(name)} {}

    reference_tests::Tensor signal;
    reference_tests::Tensor window;
    reference_tests::Tensor frame_size;
    reference_tests::Tensor frame_step;
    bool transpose_frames;

    reference_tests::Tensor expected_tensor;
    std::string test_case_name;
};

class ReferenceSTFT : public testing::TestWithParam<STFTParams>, public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.signal.data, params.window.data, params.frame_size.data, params.frame_step.data};
        refOutData = {params.expected_tensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<STFTParams>& obj) {
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
        name << "_transpose_frames_";
        name << obj.param.transpose_frames;
        return name.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const STFTParams& params) {
        const auto in_signal = std::make_shared<ov::op::v0::Parameter>(params.signal.type, params.signal.shape);
        const auto in_window = std::make_shared<ov::op::v0::Parameter>(params.window.type, params.window.shape);
        const auto in_frame_size =
            std::make_shared<ov::op::v0::Parameter>(params.frame_size.type, params.frame_size.shape);
        const auto in_frame_step =
            std::make_shared<ov::op::v0::Parameter>(params.frame_step.type, params.frame_step.shape);

        const auto STFT = std::make_shared<ov::op::v15::STFT>(in_signal,
                                                              in_window,
                                                              in_frame_size,
                                                              in_frame_step,
                                                              params.transpose_frames);
        return std::make_shared<ov::Model>(STFT->outputs(),
                                           ov::ParameterVector{in_signal, in_window, in_frame_size, in_frame_step});
    }
};

template <ov::element::Type_t ET, ov::element::Type_t IT = ov::element::i64>
std::vector<STFTParams> generateSTFTParams() {
    using VT = typename ov::element_type_traits<ET>::value_type;
    using INT_T = typename ov::element_type_traits<IT>::value_type;

    const ov::Shape signal_48_shape{1, 48};
    const ov::Shape signal_256_shape{1, 256};

    reference_tests::Tensor signal_48(
        signal_48_shape,
        ET,
        std::vector<VT>{-0.41676, -0.05627, -2.1362,  1.64027,  -1.79344, -0.84175, 0.50288,  -1.24529,
                        -1.05795, -0.90901, 0.55145,  2.29221,  0.04154,  -1.11793, 0.53906,  -0.59616,
                        -0.01913, 1.175,    -0.74787, 0.00903,  -0.87811, -0.15643, 0.25657,  -0.98878,
                        -0.33882, -0.23618, -0.63766, -1.18761, -1.42122, -0.1535,  -0.26906, 2.23137,
                        -2.43477, 0.11273,  0.37044,  1.35963,  0.50186,  -0.84421, 0.00001,  0.54235,
                        -0.31351, 0.77101,  -1.86809, 1.73118,  1.46768,  -0.33568, 0.61134,  0.04797});

    reference_tests::Tensor hann_window_5(Shape{5}, ET, std::vector<VT>{0., 0.5, 1., 0.5, 0.});
    reference_tests::Tensor hann_window_7(Shape{7}, ET, std::vector<VT>{0., 0.25, 0.75, 1., 0.75, 0.25, 0.});
    reference_tests::Tensor hann_window_8(
        Shape{8},
        ET,
        std::vector<VT>{0., 0.18826, 0.61126, 0.95048, 0.95048, 0.61126, 0.18826, 0.});
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
    reference_tests::Tensor frame_size_16(Shape{}, IT, std::vector<INT_T>{16});

    reference_tests::Tensor frame_step_2(Shape{}, IT, std::vector<INT_T>{2});
    reference_tests::Tensor frame_step_4(Shape{}, IT, std::vector<INT_T>{4});
    reference_tests::Tensor frame_step_8(Shape{}, IT, std::vector<INT_T>{8});
    reference_tests::Tensor frame_step_16(Shape{}, IT, std::vector<INT_T>{16});

    constexpr bool transpose_frames_true = true;
    constexpr bool transpose_frames_false = false;

    reference_tests::Tensor output_1_9_9_2_no_transp(
        Shape{1, 9, 9, 2},
        ET,
        std::vector<VT>{-2.52411, 0.,       0.37692,  0.,       0.0961,   0.,       -1.5093,  0.,       -3.6289,
                        0.,       -3.74213, 0.,       -0.70616, 0.,       1.11032,  0.,       1.1366,   0.,
                        1.99743,  2.45799,  -1.54106, 0.09127,  0.28179,  -0.45332, 1.11013,  -0.7782,  1.84867,
                        -0.67991, 2.6774,   1.62635,  -0.28718, 1.64988,  -1.1681,  -0.22631, 0.26235,  0.25725,
                        -2.243,   -1.74288, 2.37601,  2.51525,  0.61596,  -1.347,   -0.93254, 0.28219,  0.39666,
                        0.60667,  -1.82867, -1.87092, -0.23187, 0.65901,  1.4487,   1.18742,  -0.73965, -0.24622,
                        2.91255,  -0.82545, -0.64918, -4.09527, -1.77914, 1.08142,  1.05098,  1.02636,  0.03844,
                        0.45931,  2.58711,  1.18649,  2.7151,   -2.88547, -0.87106, -3.27669, -1.29728, -1.50822,
                        -2.56084, 2.24181,  -1.50375, 3.04262,  -0.32836, 0.0481,   -0.57373, -1.39547, -0.92956,
                        -1.32518, -1.7139,  0.02014,  -2.94864, 2.71058,  -1.16639, 2.97336,  1.78749,  1.94867,
                        0.87525,  0.70978,  1.77224,  -2.64294, 1.56859,  2.0249,   -0.08561, -0.25032, 0.47508,
                        1.29318,  -0.95946, -0.61456, 2.29916,  -1.85519, 0.52933,  -0.27895, -0.18799, 0.98232,
                        2.10241,  -2.57882, -1.11208, 2.55729,  -0.09293, -3.2837,  -0.54924, 2.25344,  0.88504,
                        -1.03814, 1.07391,  -1.06812, -3.36146, 1.49043,  2.44995,  0.58978,  -1.44897, -2.97866,
                        -1.59965, -0.02599, 0.25367,  -0.76703, 0.00445,  1.79544,  1.39855,  -1.6289,  -1.02171,
                        0.17824,  1.31769,  2.44158,  4.90558,  -1.15294, -0.47567, -2.17385, 2.46326,  1.82815,
                        -0.44417, 0.,       0.4314,   0.,       -0.63682, 0.,       -1.3278,  0.,       0.24368,
                        0.,       -2.56606, 0.,       -5.47521, 0.,       -2.60384, 0.,       -2.81501, 0.});

    reference_tests::Tensor output_1_9_5_2_no_transp(
        Shape{1, 9, 5, 2},
        ET,
        std::vector<VT>{-2.49209, 0.,       0.11167,  0.,       -1.39889, 0.,       -0.24805, 0.,       0.1782,
                        0.,       2.38232,  0.24277,  -0.17477, 0.35359,  1.30896,  0.13583,  0.36915,  -0.59295,
                        -0.3042,  -0.11877, -2.12336, -0.35253, 0.23782,  -0.58706, -1.06786, -0.30642, -0.74536,
                        1.19633,  0.51216,  0.27057,  1.81552,  0.37559,  -0.10082, 0.77398,  0.73575,  0.52785,
                        1.38254,  -1.74123, -0.47514, -0.35034, -1.41677, -0.46952, -0.2069,  -1.07442, -0.35883,
                        -0.766,   -2.21948, 2.08087,  0.05369,  0.20314,  0.77715,  0.66349,  0.40407,  1.43908,
                        -0.03963, 0.91941,  3.11192,  -2.07244, 0.57377,  0.14702,  0.11223,  -0.75968, -0.27419,
                        -1.52765, 0.42377,  -0.86016, -3.88305, 1.66473,  -1.10813, -0.4328,  -0.95271, 0.53067,
                        -0.05574, 1.01868,  -0.7169,  0.52739,  4.39323,  -0.92417, 1.39751,  0.37859,  1.30337,
                        0.,       0.2294,   0.,       0.82838,  0.,       -4.56982, 0.,       -1.47752, 0.});

    reference_tests::Tensor output_1_9_3_2_no_transp(
        Shape{1, 9, 3, 2},
        ET,
        std::vector<VT>{-2.52411, 0.,       -3.6289, 0.,       1.1366,   0.,       1.99743,  2.45799,  1.84867,
                        -0.67991, 0.26235,  0.25725, -2.243,   -1.74288, 0.39666,  0.60667,  -0.73965, -0.24622,
                        2.91255,  -0.82545, 0.03844, 0.45931,  -1.29728, -1.50822, -2.56084, 2.24181,  -0.92956,
                        -1.32518, 1.78749,  1.94867, 0.87525,  0.70978,  0.47508,  1.29318,  -0.18799, 0.98232,
                        2.10241,  -2.57882, 0.88504, -1.03814, -1.44897, -2.97866, -1.59965, -0.02599, -1.02171,
                        0.17824,  2.46326,  1.82815, -0.44417, 0.,       0.24368,  0.,       -2.81501, 0.});

    std::vector<STFTParams> params;
    params.emplace_back(signal_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_16,
                        transpose_frames_true,
                        output_1_9_3_2_no_transp,
                        "equal_size_step");
    params.emplace_back(signal_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_4,
                        transpose_frames_true,
                        output_1_9_9_2_no_transp,
                        "step_1/4_frame");
    params.emplace_back(signal_48,
                        hann_window_8,
                        frame_size_16,
                        frame_step_8,
                        transpose_frames_true,
                        output_1_9_5_2_no_transp,
                        "win_size_smaller_than_frame_size");
    return params;
}

std::vector<STFTParams> generateSTFTParams() {
    std::vector<std::vector<STFTParams>> combo_params{generateSTFTParams<ov::element::f32>()};
    std::vector<STFTParams> test_params;
    for (auto& params : combo_params)
        std::move(params.begin(), params.end(), std::back_inserter(test_params));
    return test_params;
}
}  // namespace

TEST_P(ReferenceSTFT, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         ReferenceSTFT,
                         ::testing::ValuesIn(generateSTFTParams()),
                         ReferenceSTFT::getTestCaseName);
