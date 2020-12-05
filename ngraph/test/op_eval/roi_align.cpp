//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/roi_align.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, roi_align_avg_pool)
{
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

    auto roi_align = make_shared<op::v3::ROIAlign>(
        data, rois, batch_indices, pooled_height, pooled_width, 2, 1.0f / 16.0f, "avg");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
                                26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
                                39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
                                52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                                65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        2.95833f, 3.20833f, 3.45833f, 3.70833f, 4.625f,   4.875f,   5.125f,   5.375f,   6.29167f,
        6.54167f, 6.79167f, 7.04167f, 27.9583f, 28.2083f, 28.4583f, 28.7083f, 29.625f,  29.875f,
        30.125f,  30.375f,  31.2917f, 31.5417f, 31.7917f, 32.0417f, 52.9583f, 53.2083f, 53.4583f,
        53.7083f, 54.625f,  54.875f,  55.125f,  55.375f,  56.2917f, 56.5417f, 56.7917f, 57.0417f,
        0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,      0.f,      0.f,      25.f,     25.f,     25.f,     25.f,     25.f,     25.f,
        25.f,     25.f,     25.f,     25.f,     25.f,     25.f,     50.f,     50.f,     50.f,
        50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,     50.f,
        7.39583f, 7.39583f, 7.42708f, 7.64583f, 9.0625f,  9.0625f,  9.09375f, 9.3125f,  10.7292f,
        10.7292f, 10.7604f, 10.9792f, 32.3958f, 32.3958f, 32.4271f, 32.6458f, 34.0625f, 34.0625f,
        34.0938f, 34.3125f, 35.7292f, 35.7292f, 35.7604f, 35.9792f, 57.3958f, 57.3958f, 57.4271f,
        57.6458f, 59.0625f, 59.0625f, 59.0938f, 59.3125f, 60.7292f, 60.7292f, 60.7604f, 60.9792f,
        4.27083f, 4.52083f, 4.77083f, 5.02083f, 5.9375f,  6.1875f,  6.4375f,  6.6875f,  7.60417f,
        7.85417f, 8.10417f, 8.35417f, 29.2708f, 29.5208f, 29.7708f, 30.0208f, 30.9375f, 31.1875f,
        31.4375f, 31.6875f, 32.6042f, 32.8542f, 33.1042f, 33.3542f, 54.2708f, 54.5208f, 54.7708f,
        55.0208f, 55.9375f, 56.1875f, 56.4375f, 56.6875f, 57.6042f, 57.8542f, 58.1042f, 58.3542f,
        6.77083f, 6.77083f, 6.77083f, 6.80208f, 8.4375f,  8.4375f,  8.4375f,  8.46875f, 10.1042f,
        10.1042f, 10.1042f, 10.1354f, 31.7708f, 31.7708f, 31.7708f, 31.8021f, 33.4375f, 33.4375f,
        33.4375f, 33.4688f, 35.1042f, 35.1042f, 35.1042f, 35.1354f, 56.7708f, 56.7708f, 56.7708f,
        56.8021f, 58.4375f, 58.4375f, 58.4375f, 58.4688f, 60.1042f, 60.1042f, 60.1042f, 60.1354f};
    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}
TEST(op_eval, roi_align_max_pool)
{
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

    auto roi_align = make_shared<op::v3::ROIAlign>(
        data, rois, batch_indices, pooled_height, pooled_width, 2, 1.0f / 16.0f, "max");

    auto f = make_shared<Function>(roi_align, ParameterVector{data, rois, batch_indices});

    std::vector<float> data_vec{0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
                                13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
                                26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
                                39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
                                52., 53., 54., 55., 56., 57., 58., 59., 60., 61., 62., 63., 64.,
                                65., 66., 67., 68., 69., 70., 71., 72., 73., 74.};

    std::vector<float> rois_vec{7.,   5.,  7.,  5., -15., -15., -15., -15., -10., 21.,
                                -10., 21., 13., 8., 13.,  8.,   -14., 19.,  -14., 19.};

    std::vector<int64_t> batch_indices_vec{0, 0, 0, 0, 0};

    auto result = make_shared<HostTensor>();

    ASSERT_TRUE(f->evaluate({result},
                            {make_host_tensor<element::Type_t::f32>(data_shape, data_vec),
                             make_host_tensor<element::Type_t::f32>(rois_shape, rois_vec),
                             make_host_tensor<element::Type_t::i64>(Shape{num_rois})}));

    std::vector<float> expected_vec{
        2.10938f,  2.95313f, 3.375f,   2.53125f,  3.35938f, 4.70313f, 5.375f,   4.03125f, 3.51563f,
        4.92188f,  5.625f,   4.21875f, 10.8984f,  15.2578f, 17.4375f, 13.0781f, 17.3568f, 24.2995f,
        27.7708f,  20.8281f, 18.1641f, 25.4297f,  29.0625f, 21.7969f, 19.6875f, 27.5625f, 31.5f,
        23.625f,   31.3542f, 43.8958f, 50.1667f,  37.625f,  32.8125f, 45.9375f, 52.5f,    39.375f,
        0.f,       0.f,      0.f,      0.f,       0.f,      0.f,      0.f,      0.f,      0.f,
        0.f,       0.f,      0.f,      25.f,      25.f,     25.f,     25.f,     25.f,     25.f,
        25.f,      25.f,     25.f,     25.f,      25.f,     25.f,     50.f,     50.f,     50.f,
        50.f,      50.f,     50.f,     50.f,      50.f,     50.f,     50.f,     50.f,     50.f,
        5.625f,    5.625f,   5.625f,   4.57031f,  8.95833f, 8.95833f, 8.95833f, 7.27865f, 9.375f,
        9.375f,    9.375f,   7.61719f, 19.6875f,  19.6875f, 19.6875f, 15.9961f, 31.3542f, 31.3542f,
        31.3542f,  25.4753f, 32.8125f, 32.8125f,  32.8125f, 26.6602f, 33.75f,   33.75f,   33.75f,
        27.4219f,  53.75f,   53.75f,   53.75f,    43.6719f, 56.25f,   56.25f,   56.25f,   45.7031f,
        4.5f,      3.9375f,  2.8125f,  3.9375f,   5.5f,     4.8125f,  3.4375f,  4.8125f,  4.58333f,
        4.01042f,  2.86458f, 3.9375f,  23.25f,    20.3438f, 14.5313f, 18.f,     28.4167f, 24.86458f,
        17.76042f, 22.f,     23.25f,   20.3437f,  14.5312f, 18.f,     42.f,     36.75f,   26.25f,
        32.0625f,  51.3333f, 44.9167f, 32.08333f, 39.1875f, 42.f,     36.75f,   26.25f,   32.0625f,
        4.375f,    4.375f,   4.375f,   4.375f,    7.70833f, 7.70833f, 7.70833f, 7.70833f, 9.375f,
        9.375f,    9.375f,   9.375f,   21.875f,   21.875f,  21.875f,  21.875f,  26.9792f, 26.9792f,
        26.9792f,  26.9792f, 32.8125f, 32.8125f,  32.8125f, 32.8125f, 40.1042f, 40.1042f, 40.1042f,
        40.1042f,  46.25f,   46.25f,   46.25f,    46.25f,   56.25f,   56.25f,   56.25f,   56.25f};
    const auto expected_shape = Shape{num_rois, C, pooled_height, pooled_width};

    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), expected_shape);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_vec, 6, 0.001));
}