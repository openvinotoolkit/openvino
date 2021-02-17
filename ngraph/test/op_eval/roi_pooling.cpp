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
#include "util/engine/interpreter_engine.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(op_eval, roi_pooling_invalid_roi_batch_id)
{
    const int H = 6;
    const int W = 6;
    const int image_size = H * W;
    const int channels = 1;
    const int num_rois = 1;

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

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_input<float>(feat_maps_shape, feat_maps_vect);
    // ROI with invalid batch id, should throw exception
    test_case.add_input<float>(rois_shape, {-1, 1, 1, 2, 3});
    test_case.add_expected_output<float>(output_shape, {2.0f});
    ASSERT_THROW(test_case.run(), ngraph::CheckFailure);
}
