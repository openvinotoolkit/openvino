// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <vector>
#include "gtest/gtest.h"
#include "runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

using Attrs = op::v6::ExperimentalDetectronROIFeatureExtractor::Attributes;
using ExperimentalROI = op::v6::ExperimentalDetectronROIFeatureExtractor;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, experimental_roi_feature_eval)
{
    Attrs attrs;
    attrs.aligned = false;
    attrs.output_size = 3;
    attrs.sampling_ratio = 2;
    attrs.pyramid_scales = {4};

    auto input = std::make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto pyramid_layer0 = std::make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 3});

    auto roi = std::make_shared<ExperimentalROI>(NodeVector{input, pyramid_layer0}, attrs);

    auto f0 =
        make_shared<Function>(OutputVector{roi->output(0)}, ParameterVector{input, pyramid_layer0});
    auto f1 =
        make_shared<Function>(OutputVector{roi->output(1)}, ParameterVector{input, pyramid_layer0});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> rois(shape_size(Shape{2, 4}));
    std::iota(rois.begin(), rois.end(), 0);

    std::vector<float> featmap(shape_size(Shape{1, 2, 2, 3}));
    std::iota(featmap.begin(), featmap.end(), 0);

    const auto output_features_shape = Shape{2, 2, 3, 3};
    const auto output_rois_shape = Shape{2, 4};

    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);

    auto output_features = backend->create_tensor(element::f32, output_features_shape);
    auto output_rois = backend->create_tensor(element::f32, output_rois_shape);

    auto backend_rois = backend->create_tensor(element::f32, Shape{2, 4});
    auto backend_featmap = backend->create_tensor(element::f32, Shape{1, 2, 2, 3});
    copy_data(backend_rois, rois);
    copy_data(backend_featmap, featmap);

    auto handle0 = backend->compile(f0);
    auto handle1 = backend->compile(f1);

    handle0->call_with_validate({output_features}, {backend_rois, backend_featmap});
    handle1->call_with_validate({output_rois}, {backend_rois, backend_featmap});

    const auto calculated_features = read_vector<float>(output_features);
    const auto calculated_rois = read_vector<float>(output_rois);

    std::cout << "Calculated features:\n    ";
    for (auto x : calculated_features)
    {
        std::cout << x << ", ";
    }
    std::cout << "\n\n";
    std::cout << "Calculated rois:\n    ";
    for (auto x : calculated_rois)
    {
        std::cout << x << ", ";
    }
    std::cout << "\n\n";

    ASSERT_TRUE(true);
}
