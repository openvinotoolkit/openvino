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
#include <numeric>
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

using Attrs = op::v6::ExperimentalDetectronGenerateProposalsSingleImage::Attributes;
using ExperimentalGP = op::v6::ExperimentalDetectronGenerateProposalsSingleImage;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_generate_proposals_eval)
{
    Attrs attrs;
    attrs.min_size = 0;
    attrs.nms_threshold = 0.699999988079071;
    attrs.post_nms_count = 6;
    attrs.pre_nms_count = 1000;

    auto im_info = std::make_shared<op::Parameter>(element::f32, Shape{3});
    auto anchors = std::make_shared<op::Parameter>(element::f32, Shape{36, 4});
    auto deltas = std::make_shared<op::Parameter>(element::f32, Shape{12, 2, 6});
    auto scores = std::make_shared<op::Parameter>(element::f32, Shape{3, 2, 6});

    auto proposals = std::make_shared<ExperimentalGP>(im_info, anchors, deltas, scores, attrs);

    auto f0 = make_shared<Function>(OutputVector{proposals->output(0)},
                                    ParameterVector{im_info, anchors, deltas, scores});
    auto f1 = make_shared<Function>(OutputVector{proposals->output(1)},
                                    ParameterVector{im_info, anchors, deltas, scores});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    std::vector<float> im_info_data = {1.0f, 1.0f, 1.0f};
    std::vector<float> anchors_data = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f};
    std::vector<float> deltas_data = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f};
    std::vector<float> scores_data = {
        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f};

    const auto output_rois_shape = Shape{6, 4};
    const auto output_scores_shape = Shape{6};

    std::vector<float> expected_output_rois = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    std::vector<float> expected_output_scores = {8.0f, 5.0f, 4.0f, 1.0f, 1.0f, 1.0f};

    auto output_rois = backend->create_tensor(element::f32, output_rois_shape);
    auto output_scores = backend->create_tensor(element::f32, output_scores_shape);

    auto backend_im_info = backend->create_tensor(element::f32, Shape{3});
    auto backend_anchors = backend->create_tensor(element::f32, Shape{36, 4});
    auto backend_deltas = backend->create_tensor(element::f32, Shape{12, 2, 6});
    auto backend_scores = backend->create_tensor(element::f32, Shape{3, 2, 6});

    copy_data(backend_im_info, im_info_data);
    copy_data(backend_anchors, anchors_data);
    copy_data(backend_deltas, deltas_data);
    copy_data(backend_scores, scores_data);

    auto handle0 = backend->compile(f0);
    auto handle1 = backend->compile(f1);

    handle0->call_with_validate({output_rois},
                                {backend_im_info, backend_anchors, backend_deltas, backend_scores});
    handle1->call_with_validate({output_scores},
                                {backend_im_info, backend_anchors, backend_deltas, backend_scores});

    auto output_rois_vector = read_vector<float>(output_rois);
    auto output_scores_vector = read_vector<float>(output_scores);
    std::cout << "Output rois:\n    ";
    for (auto x : output_rois_vector)
    {
        std::cout << x << ", ";
    }
    std::cout << "\n\n";
    std::cout << "Output scores:\n    ";
    for (auto x : output_scores_vector)
    {
        std::cout << x << ", ";
    }
    std::cout << "\n\n";

    EXPECT_TRUE(test::all_close_f(
        expected_output_rois, read_vector<float>(output_rois), MIN_FLOAT_TOLERANCE_BITS));
    EXPECT_TRUE(test::all_close_f(
        expected_output_scores, read_vector<float>(output_scores), MIN_FLOAT_TOLERANCE_BITS));

//    ASSERT_TRUE(true);
}
