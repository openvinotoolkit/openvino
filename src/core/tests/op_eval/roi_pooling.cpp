// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(op_eval, roi_pooling_invalid_roi_batch_id) {
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
    const auto roi_pooling = make_shared<op::v0::ROIPooling>(feat_maps, rois, pooled_shape, spatial_scale, "max");
    const auto f = make_shared<Function>(roi_pooling, ParameterVector{feat_maps, rois});

    vector<float> feat_maps_vect;
    for (unsigned int i = 0; i < channels * image_size; i++) {
        feat_maps_vect.push_back(1.f * i / 10);
    }

    auto test_case = test::TestCase(f, "TEMPLATE");
    test_case.add_input<float>(feat_maps_shape, feat_maps_vect);
    // ROI with invalid batch id, should throw exception
    test_case.add_input<float>(rois_shape, {-1, 1, 1, 2, 3});
    test_case.add_expected_output<float>(output_shape, {2.0f});
    ASSERT_ANY_THROW(test_case.run());
}
