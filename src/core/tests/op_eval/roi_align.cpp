// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/roi_align.hpp"

#include <numeric>
#include <vector>

#include "engines_util/execute_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

OPENVINO_SUPPRESS_DEPRECATED_START

TEST(op_eval, roi_align_avg_pool) {
    const int N = 1;
    const int C = 3;
    const int H = 5;
    const int W = 5;
    const int num_rois = 5;
    const int pooled_height = 3;
    const int pooled_width = 4;
    const auto data_shape = Shape{N, C, H, W};
    const auto rois_shape = Shape{num_rois, 4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{num_rois});

    auto roi_align =
        make_shared<op::v3::ROIAlign>(data, rois, batch_indices, pooled_height, pooled_width, 2, 1.0f / 16.0f, "avg");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f, 6.54167f, 6.79167f,
        7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,  30.125f,  30.375f,  31.2917f, 31.5417f,
        31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f, 53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f,
        56.5417f, 56.7917f, 57.0417f, 0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
        25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
        50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,
        9.0625f,  9.09375f, 9.3125f,  10.7292f, 10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f,
        34.0625f, 34.0625f, 34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
        57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f, 4.27083f, 4.52083f,
        4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f, 7.85417f, 8.10417f, 8.35417f, 29.2708f,
        29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f, 31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f,
        54.2708f, 54.5208f, 54.7708f, 55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f,
        58.3542f, 6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f, 10.1042f,
        10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f, 33.4375f, 33.4688f, 35.1042f,
        35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f, 56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f,
        60.1042f, 60.1042f, 60.1042f, 60.1354f};
    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
TEST(op_eval, roi_align_max_pool) {
    const int N = 1;
    const int C = 3;
    const int H = 5;
    const int W = 5;
    const int num_rois = 5;
    const int pooled_height = 3;
    const int pooled_width = 4;
    const auto data_shape = Shape{N, C, H, W};
    const auto rois_shape = Shape{num_rois, 4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{num_rois});

    auto roi_align =
        make_shared<op::v3::ROIAlign>(data, rois, batch_indices, pooled_height, pooled_width, 2, 1.0f / 16.0f, "max");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        3.4375,  3.6875,  3.9375,  4.1875,  5.10417, 5.35417, 5.60417, 5.85417, 6.77083, 7.02083, 7.27083, 7.52083,
        28.4375, 28.6875, 28.9375, 29.1875, 30.1042, 30.3542, 30.6042, 30.8542, 31.7708, 32.0208, 32.2708, 32.5208,
        53.4375, 53.6875, 53.9375, 54.1875, 55.1042, 55.3542, 55.6042, 55.8542, 56.7708, 57.0208, 57.2708, 57.5208,
        0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,
        25,      25,      25,      25,      25,      25,      25,      25,      25,      25,      25,      25,
        50,      50,      50,      50,      50,      50,      50,      50,      50,      50,      50,      50,
        7.8125,  7.8125,  7.875,   8.125,   9.47917, 9.47917, 9.54167, 9.79167, 11.1458, 11.1458, 11.2083, 11.4583,
        32.8125, 32.8125, 32.875,  33.125,  34.4792, 34.4792, 34.5417, 34.7917, 36.1458, 36.1458, 36.2083, 36.4583,
        57.8125, 57.8125, 57.875,  58.125,  59.4792, 59.4792, 59.5417, 59.7917, 61.1458, 61.1458, 61.2083, 61.4583,
        4.75,    5,       5.25,    5.5,     6.41667, 6.66667, 6.91667, 7.16667, 8.08333, 8.33333, 8.58333, 8.83333,
        29.75,   30,      30.25,   30.5,    31.4167, 31.6667, 31.9167, 32.1667, 33.0833, 33.3333, 33.5833, 33.8333,
        54.75,   55,      55.25,   55.5,    56.4167, 56.6667, 56.9167, 57.1667, 58.0833, 58.3333, 58.5833, 58.8333,
        7.1875,  7.1875,  7.1875,  7.25,    8.85417, 8.85417, 8.85417, 8.91667, 10.5208, 10.5208, 10.5208, 10.5833,
        32.1875, 32.1875, 32.1875, 32.25,   33.8542, 33.8542, 33.8542, 33.9167, 35.5208, 35.5208, 35.5208, 35.5833,
        57.1875, 57.1875, 57.1875, 57.25,   58.8542, 58.8542, 58.8542, 58.9167, 60.5208, 60.5208, 60.5208, 60.5833};
    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
TEST(op_eval, roi_align_9_avg_pool_asymmetric) {
    const int N = 1;
    const int C = 3;
    const int H = 5;
    const int W = 5;
    const int num_rois = 5;
    const int pooled_height = 3;
    const int pooled_width = 4;
    const auto data_shape = Shape{N, C, H, W};
    const auto rois_shape = Shape{num_rois, 4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{num_rois});

    auto roi_align = make_shared<op::v9::ROIAlign>(data,
                                                   rois,
                                                   batch_indices,
                                                   pooled_height,
                                                   pooled_width,
                                                   2,
                                                   1.0f / 16.0f,
                                                   "avg",
                                                   "asymmetric");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f, 6.54167f, 6.79167f,
        7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,  30.125f,  30.375f,  31.2917f, 31.5417f,
        31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f, 53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f,
        56.5417f, 56.7917f, 57.0417f, 0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
        25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
        50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,
        9.0625f,  9.09375f, 9.3125f,  10.7292f, 10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f,
        34.0625f, 34.0625f, 34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
        57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f, 4.27083f, 4.52083f,
        4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f, 7.85417f, 8.10417f, 8.35417f, 29.2708f,
        29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f, 31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f,
        54.2708f, 54.5208f, 54.7708f, 55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f,
        58.3542f, 6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f, 10.1042f,
        10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f, 33.4375f, 33.4688f, 35.1042f,
        35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f, 56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f,
        60.1042f, 60.1042f, 60.1042f, 60.1354f};

    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
TEST(op_eval, roi_align_9_avg_pool_default_aligned_mode) {
    const int N = 1;
    const int C = 3;
    const int H = 5;
    const int W = 5;
    const int num_rois = 5;
    const int pooled_height = 3;
    const int pooled_width = 4;
    const auto data_shape = Shape{N, C, H, W};
    const auto rois_shape = Shape{num_rois, 4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{num_rois});

    auto roi_align =
        make_shared<op::v9::ROIAlign>(data, rois, batch_indices, pooled_height, pooled_width, 2, 1.0f / 16.0f, "avg");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f, 6.54167f, 6.79167f,
        7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,  30.125f,  30.375f,  31.2917f, 31.5417f,
        31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f, 53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f,
        56.5417f, 56.7917f, 57.0417f, 0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
        25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
        50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,
        9.0625f,  9.09375f, 9.3125f,  10.7292f, 10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f,
        34.0625f, 34.0625f, 34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
        57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f, 4.27083f, 4.52083f,
        4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f, 7.85417f, 8.10417f, 8.35417f, 29.2708f,
        29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f, 31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f,
        54.2708f, 54.5208f, 54.7708f, 55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f,
        58.3542f, 6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f, 10.1042f,
        10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f, 33.4375f, 33.4688f, 35.1042f,
        35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f, 56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f,
        60.1042f, 60.1042f, 60.1042f, 60.1354f};

    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
TEST(op_eval, roi_align_9_avg_pool_tf_half_pixel_for_nn) {
    const int N = 1;
    const int C = 3;
    const int H = 5;
    const int W = 5;
    const int num_rois = 5;
    const int pooled_height = 3;
    const int pooled_width = 4;
    const auto data_shape = Shape{N, C, H, W};
    const auto rois_shape = Shape{num_rois, 4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{num_rois});

    auto roi_align = make_shared<op::v9::ROIAlign>(data,
                                                   rois,
                                                   batch_indices,
                                                   pooled_height,
                                                   pooled_width,
                                                   2,
                                                   1.0f / 16.0f,
                                                   "avg",
                                                   "tf_half_pixel_for_nn");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
        25.f,     25.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
        50.f,     50.f,     50.f,     0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.3125f,  0.3125f,
        0.3125f,  0.3125f,  0.3125f,  0.3125f,  0.3125f,  0.3125f,  0.3125f,  0.3125f,  0.3125f,  0.3125f,  25.3125f,
        25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f, 25.3125f,
        50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f, 50.3125f,
        50.3125f, 0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      0.f};

    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
TEST(op_eval, roi_align_9_max_pool_half_pixel) {
    const int N = 1;
    const int C = 3;
    const int H = 5;
    const int W = 5;
    const int num_rois = 5;
    const int pooled_height = 3;
    const int pooled_width = 4;
    const auto data_shape = Shape{N, C, H, W};
    const auto rois_shape = Shape{num_rois, 4};

    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto batch_indices = make_shared<op::Parameter>(element::i32, Shape{num_rois});

    auto roi_align = make_shared<op::v9::ROIAlign>(data,
                                                   rois,
                                                   batch_indices,
                                                   pooled_height,
                                                   pooled_width,
                                                   2,
                                                   1.0f / 16.0f,
                                                   "max",
                                                   "half_pixel");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12., 13., 14.,
                                15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.,
                                30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44.,
                                45., 46., 47., 48., 49., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59.,
                                60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   25.f,  25.f,  25.f,
        25.f,  25.f,  25.f,  25.f,  25.f,  25.f,  25.f,  25.f,  25.f,  50.f,  50.f,  50.f,  50.f,  50.f,  50.f,
        50.f,  50.f,  50.f,  50.f,  50.f,  50.f,  0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
        25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 25.5f, 50.5f, 50.5f, 50.5f,
        50.5f, 50.5f, 50.5f, 50.5f, 50.5f, 50.5f, 50.5f, 50.5f, 50.5f, 0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,
        0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f,   0.f};
    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
