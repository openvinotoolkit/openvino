//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, roi_pooling_1x1_max)
{
    const int H = 6;
    const int W = 6;
    const int image_size = H * W;
    const int channels = 3;
    const int num_rois = 3;

    const int pooled_h = 1;
    const int pooled_w = 1;
    const float spatial_scale = 1.f;

    Shape feat_maps_shape{1, channels, H, W};
    Shape rois_shape{num_rois, 5};
    Shape pooled_shape{pooled_h, pooled_w};
    Shape output_shape{num_rois, channels, pooled_h, pooled_w};

    const auto feat_maps = make_shared<op::Parameter>(element::f32, feat_maps_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto roi_pooling =
        make_shared<op::v0::ROIPooling>(feat_maps, rois, pooled_shape, spatial_scale, "max");
    const auto f = make_shared<Function>(roi_pooling, ParameterVector{feat_maps, rois});

    vector<float> feat_maps_vect;
    for (unsigned int i = 0; i < channels * image_size; i++)
    {
        feat_maps_vect.push_back(1.f * i / 10);
    }

    vector<float> rois_vect = {0, 1, 1, 2, 3, 0, 1, 1, 2, 3, 0, 1, 1, 2, 3};

    const vector<float> expected_vect = {2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f, 2.0f, 5.6f, 9.2f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(feat_maps_shape, feat_maps_vect);
    test_case.add_input<float>(rois_shape, rois_vect);
    test_case.add_expected_output<float>(output_shape, expected_vect);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, roi_pooling_2x2_max)
{
    const int H = 6;
    const int W = 6;
    const int image_size = H * W;
    const int channels = 1;
    const int num_rois = 3;

    const int pooled_h = 2;
    const int pooled_w = 2;
    const float spatial_scale = 1.f;

    Shape feat_maps_shape{1, channels, H, W};
    Shape rois_shape{num_rois, 5};
    Shape pooled_shape{pooled_h, pooled_w};
    Shape output_shape{num_rois, channels, pooled_h, pooled_w};

    const auto feat_maps = make_shared<op::Parameter>(element::f32, feat_maps_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto roi_pooling =
        make_shared<op::v0::ROIPooling>(feat_maps, rois, pooled_shape, spatial_scale, "max");
    const auto f = make_shared<Function>(roi_pooling, ParameterVector{feat_maps, rois});

    vector<float> feat_maps_vect;
    for (unsigned int i = 0; i < channels * image_size; i++)
    {
        feat_maps_vect.push_back(1.f * i / 10);
    }

    vector<float> rois_vect = {0, 1, 1, 3, 3, 0, 1, 2, 2, 4, 0, 0, 1, 4, 5};

    const vector<float> expected_vect = {
        1.4f, 1.5f, 2.0f, 2.1f, 1.9f, 2.0f, 2.5f, 2.6f, 2.0f, 2.2f, 3.2f, 3.4f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(feat_maps_shape, feat_maps_vect);
    test_case.add_input<float>(rois_shape, rois_vect);
    test_case.add_expected_output<float>(output_shape, expected_vect);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, roi_pooling_1x1_bilinear)
{
    const int H = 6;
    const int W = 6;
    const int image_size = H * W;
    const int channels = 3;
    const int num_rois = 2;

    const int pooled_h = 1;
    const int pooled_w = 1;
    const float spatial_scale = 1.f;

    Shape feat_maps_shape{1, channels, H, W};
    Shape rois_shape{num_rois, 5};
    Shape pooled_shape{pooled_h, pooled_w};
    Shape output_shape{num_rois, channels, pooled_h, pooled_w};

    const auto feat_maps = make_shared<op::Parameter>(element::f32, feat_maps_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto roi_pooling =
        make_shared<op::v0::ROIPooling>(feat_maps, rois, pooled_shape, spatial_scale, "bilinear");
    const auto f = make_shared<Function>(roi_pooling, ParameterVector{feat_maps, rois});

    vector<float> feat_maps_vect;
    for (unsigned int i = 0; i < channels * image_size; i++)
    {
        feat_maps_vect.push_back(1.f * i / 10);
    }

    vector<float> rois_vect = {0, 0.2, 0.2, 0.4, 0.4, 0, 0.2, 0.2, 0.6, 0.6};

    const vector<float> expected_vect = {1.05f, 4.65f, 8.25f, 1.4f, 5.0f, 8.6f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(feat_maps_shape, feat_maps_vect);
    test_case.add_input<float>(rois_shape, rois_vect);
    test_case.add_expected_output<float>(output_shape, expected_vect);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, roi_pooling_2x2_bilinear)
{
    const int H = 8;
    const int W = 8;
    const int image_size = H * W;
    const int channels = 1;
    const int num_rois = 3;

    const int pooled_h = 2;
    const int pooled_w = 2;
    const float spatial_scale = 1.f;

    Shape feat_maps_shape{1, channels, H, W};
    Shape rois_shape{num_rois, 5};
    Shape pooled_shape{pooled_h, pooled_w};
    Shape output_shape{num_rois, channels, pooled_h, pooled_w};

    const auto feat_maps = make_shared<op::Parameter>(element::f32, feat_maps_shape);
    const auto rois = make_shared<op::Parameter>(element::f32, rois_shape);
    const auto roi_pooling =
        make_shared<op::v0::ROIPooling>(feat_maps, rois, pooled_shape, spatial_scale, "bilinear");
    const auto f = make_shared<Function>(roi_pooling, ParameterVector{feat_maps, rois});

    vector<float> feat_maps_vect;
    for (unsigned int i = 0; i < channels * image_size; i++)
    {
        feat_maps_vect.push_back(1.f * i / 10);
    }

    vector<float> rois_vect = {0.f,
                               0.15f,
                               0.2f,
                               0.75f,
                               0.8f,
                               0.f,
                               0.15f,
                               0.2f,
                               0.75f,
                               0.8f,
                               0.f,
                               0.15f,
                               0.2f,
                               0.75f,
                               0.8f};

    const auto count = shape_size(output_shape);
    const vector<float> expected_vect = {1.225f,
                                         1.645f,
                                         4.585f,
                                         5.005f,
                                         1.225f,
                                         1.645f,
                                         4.585f,
                                         5.005f,
                                         1.225f,
                                         1.645f,
                                         4.585f,
                                         5.005f};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>(feat_maps_shape, feat_maps_vect);
    test_case.add_input<float>(rois_shape, rois_vect);
    test_case.add_expected_output<float>(output_shape, expected_vect);
    test_case.run();
}
