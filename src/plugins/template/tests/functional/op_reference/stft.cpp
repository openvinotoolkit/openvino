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

    const ov::Shape signal_48_shape{48};
    const ov::Shape signal_1_48_shape{1, 48};
    const ov::Shape signal_2_48_shape{2, 48};
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

    reference_tests::Tensor signal_1_48(
        signal_1_48_shape,
        ET,
        std::vector<VT>{-0.41676, -0.05627, -2.1362,  1.64027,  -1.79344, -0.84175, 0.50288,  -1.24529,
                        -1.05795, -0.90901, 0.55145,  2.29221,  0.04154,  -1.11793, 0.53906,  -0.59616,
                        -0.01913, 1.175,    -0.74787, 0.00903,  -0.87811, -0.15643, 0.25657,  -0.98878,
                        -0.33882, -0.23618, -0.63766, -1.18761, -1.42122, -0.1535,  -0.26906, 2.23137,
                        -2.43477, 0.11273,  0.37044,  1.35963,  0.50186,  -0.84421, 0.00001,  0.54235,
                        -0.31351, 0.77101,  -1.86809, 1.73118,  1.46768,  -0.33568, 0.61134,  0.04797});

    reference_tests::Tensor signal_2_48(
        signal_2_48_shape,
        ET,
        std::vector<VT>{-0.41676, -0.05627, -2.1362,  1.64027,  -1.79344, -0.84175, 0.50288,  -1.24529, -1.05795,
                        -0.90901, 0.55145,  2.29221,  0.04154,  -1.11793, 0.53906,  -0.59616, -0.01913, 1.175,
                        -0.74787, 0.00903,  -0.87811, -0.15643, 0.25657,  -0.98878, -0.33882, -0.23618, -0.63766,
                        -1.18761, -1.42122, -0.1535,  -0.26906, 2.23137,  -2.43477, 0.11273,  0.37044,  1.35963,
                        0.50186,  -0.84421, 0.00001,  0.54235,  -0.31351, 0.77101,  -1.86809, 1.73118,  1.46768,
                        -0.33568, 0.61134,  0.04797,  -0.82914, 0.08771,  1.00037,  -0.38109, -0.37567, -0.07447,
                        0.4335,   1.27838,  -0.63468, 0.5084,   0.21612,  -1.85861, -0.41932, -0.13233, -0.03957,
                        0.326,    -2.04032, 0.04626,  -0.67768, -1.43944, 0.5243,   0.73528,  -0.65325, 0.84246,
                        -0.38152, 0.06649,  -1.09874, 1.58449,  -2.65945, -0.09145, 0.69512,  -2.03347, -0.18947,
                        -0.07722, 0.8247,   1.24821,  -0.40389, -1.38452, 1.36724,  1.21789,  -0.46201, 0.35089,
                        0.38187,  0.56628,  0.20421,  1.4067,   -1.73796, 1.04082});

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

    constexpr bool transpose_frames_true = true;
    constexpr bool transpose_frames_false = false;

    reference_tests::Tensor output_1_9_9_2_transp(
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

    reference_tests::Tensor output_2_9_3_2_transp(
        Shape{2, 9, 3, 2},
        ET,
        std::vector<VT>{-2.52411, 0.,       -3.6289,  0.,       1.1366,   0.,       1.99743,  2.45799,  1.84867,
                        -0.67991, 0.26235,  0.25725,  -2.243,   -1.74288, 0.39666,  0.60667,  -0.73965, -0.24622,
                        2.91255,  -0.82545, 0.03844,  0.45931,  -1.29728, -1.50822, -2.56084, 2.24181,  -0.92956,
                        -1.32518, 1.78749,  1.94867,  0.87525,  0.70978,  0.47508,  1.29318,  -0.18799, 0.98232,
                        2.10241,  -2.57882, 0.88504,  -1.03814, -1.44897, -2.97866, -1.59965, -0.02599, -1.02171,
                        0.17824,  2.46326,  1.82815,  -0.44417, 0.,       0.24368,  0.,       -2.81501, 0.,
                        0.23009,  0.,       -0.69414, 0.,       2.43185,  0.,       -0.8824,  -1.32292, -0.24572,
                        -0.82491, -1.45408, 0.19868,  1.82039,  1.39297,  0.23871,  1.03274,  0.813,    0.27265,
                        -0.61264, -0.76088, 0.88512,  0.11954,  -0.37696, -2.37281, -1.69806, -0.27959, 0.49037,
                        0.61252,  -2.19384, 2.44019,  2.42024,  -0.07393, -2.01537, -2.16847, 3.35813,  -0.14251,
                        -2.3712,  1.26736,  0.26513,  0.28205,  -1.42191, -1.20478, 1.58578,  -0.88636, 2.88537,
                        1.72055,  0.30074,  1.25455,  -0.75431, 0.,       -4.31307, 0.,       -0.48201, 0.});

    reference_tests::Tensor output_2_3_9_2_no_transp(
        Shape{2, 3, 9, 2},
        ET,
        std::vector<VT>{
            -2.52411, 0.,      1.99743,  2.45799,  -2.243,   -1.74288, 2.91255,  -0.82545, -2.56084, 2.24181,  0.87525,
            0.70978,  2.10241, -2.57882, -1.59965, -0.02599, -0.44417, 0.,       -3.6289,  0.,       1.84867,  -0.67991,
            0.39666,  0.60667, 0.03844,  0.45931,  -0.92956, -1.32518, 0.47508,  1.29318,  0.88504,  -1.03814, -1.02171,
            0.17824,  0.24368, 0.,       1.1366,   0.,       0.26235,  0.25725,  -0.73965, -0.24622, -1.29728, -1.50822,
            1.78749,  1.94867, -0.18799, 0.98232,  -1.44897, -2.97866, 2.46326,  1.82815,  -2.81501, 0.,       0.23009,
            0.,       -0.8824, -1.32292, 1.82039,  1.39297,  -0.61264, -0.76088, -1.69806, -0.27959, 2.42024,  -0.07393,
            -2.3712,  1.26736, 1.58578,  -0.88636, -0.75431, 0.,       -0.69414, 0.,       -0.24572, -0.82491, 0.23871,
            1.03274,  0.88512, 0.11954,  0.49037,  0.61252,  -2.01537, -2.16847, 0.26513,  0.28205,  2.88537,  1.72055,
            -4.31307, 0.,      2.43185,  0.,       -1.45408, 0.19868,  0.813,    0.27265,  -0.37696, -2.37281, -2.19384,
            2.44019,  3.35813, -0.14251, -1.42191, -1.20478, 0.30074,  1.25455,  -0.48201, 0.});

    reference_tests::Tensor output_1_9_9_2_transp_win_pad(
        Shape{1, 9, 9, 2},
        ET,
        std::vector<VT>{-2.49209, 0.,       1.80228,  0.,       0.11167,  0.,       -1.10931, 0.,       -1.39889,
                        0.,       -3.05836, 0.,       -0.24805, 0.,       1.50095,  0.,       0.1782,   0.,
                        2.38232,  0.24277,  -1.66564, -1.10376, -0.17477, 0.35359,  1.11949,  0.11317,  1.30896,
                        0.13583,  2.80885,  0.67694,  0.36915,  -0.59295, -1.36243, -0.87174, -0.3042,  -0.11877,
                        -2.12336, -0.35253, 1.21787,  2.13837,  0.23782,  -0.58706, -1.05259, -0.27535, -1.06786,
                        -0.30642, -2.18394, -1.1024,  -0.74536, 1.19633,  1.01091,  1.52014,  0.51216,  0.27057,
                        1.81552,  0.37559,  -0.45972, -2.87627, -0.10082, 0.77398,  0.7832,   0.34578,  0.73575,
                        0.52785,  1.46622,  1.17898,  1.38254,  -1.74123, -0.59435, -1.82269, -0.47514, -0.35034,
                        -1.41677, -0.46952, -0.39908, 3.03318,  -0.2069,  -1.07442, -0.42578, -0.117,   -0.35883,
                        -0.766,   -0.91042, -0.99052, -2.21948, 2.08087,  0.25057,  1.78712,  0.05369,  0.20314,
                        0.77715,  0.66349,  1.00098,  -2.54308, 0.40407,  1.43908,  0.30786,  -0.36902, -0.03963,
                        0.91941,  0.61261,  0.69939,  3.11192,  -2.07244, -0.03943, -1.50246, 0.57377,  0.14702,
                        0.11223,  -0.75968, -1.1389,  1.66717,  -0.27419, -1.52765, -0.61667, 0.73554,  0.42377,
                        -0.86016, -0.51775, -0.42416, -3.88305, 1.66473,  -0.0569,  1.06726,  -1.10813, -0.4328,
                        -0.95271, 0.53067,  0.96646,  -0.77057, -0.05574, 1.01868,  1.12796,  -0.60164, -0.7169,
                        0.52739,  0.51569,  0.19735,  4.39323,  -0.92417, 0.08818,  -0.55152, 1.39751,  0.37859,
                        1.30337,  0.,       -0.84619, 0.,       0.2294,   0.,       -1.37763, 0.,       0.82838,
                        0.,       -0.52417, 0.,       -4.56982, 0.,       -0.09405, 0.,       -1.47752, 0.});

    reference_tests::Tensor output_1_9_5_2_transp(
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

    reference_tests::Tensor output_9_3_2_transp(
        Shape{9, 3, 2},
        ET,
        std::vector<VT>{-2.52411, 0.,       -3.6289, 0.,       1.1366,   0.,       1.99743,  2.45799,  1.84867,
                        -0.67991, 0.26235,  0.25725, -2.243,   -1.74288, 0.39666,  0.60667,  -0.73965, -0.24622,
                        2.91255,  -0.82545, 0.03844, 0.45931,  -1.29728, -1.50822, -2.56084, 2.24181,  -0.92956,
                        -1.32518, 1.78749,  1.94867, 0.87525,  0.70978,  0.47508,  1.29318,  -0.18799, 0.98232,
                        2.10241,  -2.57882, 0.88504, -1.03814, -1.44897, -2.97866, -1.59965, -0.02599, -1.02171,
                        0.17824,  2.46326,  1.82815, -0.44417, 0.,       0.24368,  0.,       -2.81501, 0.});

    reference_tests::Tensor output_1_9_3_2_transp(
        Shape{1, 9, 3, 2},
        ET,
        std::vector<VT>{-2.52411, 0.,       -3.6289, 0.,       1.1366,   0.,       1.99743,  2.45799,  1.84867,
                        -0.67991, 0.26235,  0.25725, -2.243,   -1.74288, 0.39666,  0.60667,  -0.73965, -0.24622,
                        2.91255,  -0.82545, 0.03844, 0.45931,  -1.29728, -1.50822, -2.56084, 2.24181,  -0.92956,
                        -1.32518, 1.78749,  1.94867, 0.87525,  0.70978,  0.47508,  1.29318,  -0.18799, 0.98232,
                        2.10241,  -2.57882, 0.88504, -1.03814, -1.44897, -2.97866, -1.59965, -0.02599, -1.02171,
                        0.17824,  2.46326,  1.82815, -0.44417, 0.,       0.24368,  0.,       -2.81501, 0.});

    reference_tests::Tensor output_1_6_13_2_transp(
        Shape{1, 6, 13, 2},
        ET,
        std::vector<VT>{
            -1.71092, 0.,       -2.41009, 0.,       2.23022,  0.,       -0.7409,  0.,       0.45297,  0.,
            -1.11149, 0.,       -1.14862, 0.,       -2.14551, 0.,       -1.16026, 0.,       -0.65135, 0.,
            1.83099,  0.,       -0.1793,  0.,       -0.2968,  0.,       1.47212,  0.71877,  2.17268,  0.79158,
            -2.28473, -0.93586, 0.4625,   0.34192,  -0.56009, -0.32899, 0.93528,  0.44276,  1.11077,  0.05564,
            1.82719,  -0.1221,  0.71587,  1.50743,  1.10802,  -0.41842, -1.71345, -0.67438, 0.05781,  0.40969,
            0.4558,   -0.24137, -0.54856, -1.56669, -1.47087, -1.22889, 2.1535,   1.84441,  0.18738,  -0.28908,
            0.66134,  0.88008,  -0.66811, -0.52077, -1.02705, -0.15929, -1.12869, 0.2893,   0.0583,   -1.66476,
            -2.16394, 0.18383,  1.42389,  1.02343,  0.32308,  -0.7337,  -0.68826, 0.55139,  -0.91886, 1.85309,
            0.52177,  0.97814,  -1.50306, -2.29021, -0.76526, -0.28515, -0.47423, -1.4385,  0.63386,  0.43591,
            0.90989,  0.38369,  0.51776,  -0.36462, -0.31809, 0.57129,  2.99689,  0.98808,  -1.06897, -0.98176,
            -0.81284, 0.72147,  0.63521,  -1.1571,  1.74128,  -1.03922, 0.14692,  -0.1082,  0.64531,  1.98433,
            0.856,    1.12631,  0.14133,  1.66429,  -0.63884, -0.57479, -0.6772,  -0.71798, -0.19529, 0.22579,
            0.09013,  0.66192,  -2.7275,  -2.70068, 0.6808,   0.74142,  0.95724,  -0.28153, -0.33733, 2.09067,
            -0.89051, -0.04374, -0.16546, -0.69762, -0.12612, -1.43585, -0.37017, -1.74231, 0.00518,  -1.6207,
            0.29356,  0.84215,  0.2579,   0.98549,  0.05179,  -0.0244,  0.03393,  -1.30044, 1.1122,   3.98255,
            -0.23778, -0.54982, -0.43563, -0.19685, 0.08299,  -2.86001});

    reference_tests::Tensor output_1_5_1_2_transp(
        Shape{1, 5, 1, 2},
        ET,
        std::vector<VT>{-1.39417, 0., 1.6707, -0.24081, -2.10633, -0.17201, 1.92729, 1.18867, -0.79458, -2.06197});

    reference_tests::Tensor output_1_13_4_2_transp(
        Shape{1, 13, 4, 2},
        ET,
        std::vector<VT>{0.37692,  0.,       -1.5093,  0.,       -3.74213, 0.,       1.11032,  0.,       -0.97443,
                        0.37675,  1.29346,  -0.61484, 3.19854,  1.21384,  -1.1273,  -0.1418,  2.06069,  0.5741,
                        -0.96908, 0.783,    -2.19122, -1.86238, 1.24662,  0.3837,   -2.37601, -2.51525, 0.93254,
                        -0.28219, 1.82867,  1.87092,  -1.4487,  -1.18742, 1.43872,  3.93066,  -1.04922, -0.61175,
                        -2.32056, -1.48206, 1.28249,  2.62385,  0.18916,  -3.90786, 0.97448,  1.3296,   2.63709,
                        0.82637,  -0.23282, -3.6278,  -1.50375, 3.04262,  -0.57373, -1.39547, -1.7139,  0.02014,
                        -1.16639, 2.97336,  1.88374,  -2.58177, 0.05181,  0.50975,  -0.1459,  -0.62862, 1.2734,
                        -1.07889, -1.56702, 2.76644,  0.04708,  1.07766,  1.43297,  0.30029,  0.57761,  -0.09787,
                        1.11208,  -2.55729, 0.54924,  -2.25344, -1.07391, 1.06812,  -2.44995, -0.58978, -0.58628,
                        1.41096,  -1.24758, 2.10349,  -0.48732, -2.3458,  1.78469,  1.94472,  -0.08106, -0.29824,
                        1.40955,  -1.06578, 1.98965,  2.02462,  1.00713,  -1.87562, 0.4314,   0.,       -1.3278,
                        0.,       -2.56606, 0.,       -2.60384, 0.});

    std::vector<STFTParams> params;
    params.emplace_back(signal_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_16,
                        transpose_frames_true,
                        output_9_3_2_transp,
                        "basic_1D_transp");
    params.emplace_back(signal_1_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_16,
                        transpose_frames_true,
                        output_1_9_3_2_transp,
                        "basic_batch_1_transp");
    params.emplace_back(signal_2_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_16,
                        transpose_frames_true,
                        output_2_9_3_2_transp,
                        "basic_batch_2_transp");
    params.emplace_back(signal_2_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_16,
                        transpose_frames_false,
                        output_2_3_9_2_no_transp,
                        "basic_batch_2_no_transp");
    params.emplace_back(signal_1_48,
                        hann_window_16,
                        frame_size_16,
                        frame_step_4,
                        transpose_frames_true,
                        output_1_9_9_2_transp,
                        "step_1/4_frame_transp");
    params.emplace_back(signal_1_48,
                        hann_window_8,
                        frame_size_16,
                        frame_step_8,
                        transpose_frames_true,
                        output_1_9_5_2_transp,
                        "win_size_<_frame_size_transp");
    params.emplace_back(signal_1_48,
                        hann_window_8,
                        frame_size_16,
                        frame_step_4,
                        transpose_frames_true,
                        output_1_9_9_2_transp_win_pad,
                        "step_1/4_frame_&_win_size_<_frame_size_transp");
    params.emplace_back(signal_1_48,
                        hann_window_7,
                        frame_size_11,
                        frame_step_3,
                        transpose_frames_true,
                        output_1_6_13_2_transp,
                        "odd_sizes_transp");
    params.emplace_back(signal_1_48,
                        hann_window_5,
                        frame_size_9,
                        frame_step_100,
                        transpose_frames_true,
                        output_1_5_1_2_transp,
                        "step_>_signal_size_transp");
    return params;
}

std::vector<STFTParams> generateSTFTParams() {
    std::vector<std::vector<STFTParams>> combo_params{generateSTFTParams<ov::element::f16>(),
                                                      generateSTFTParams<ov::element::f32>()};
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
