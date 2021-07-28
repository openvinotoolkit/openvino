// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/engine/test_engines.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

using ExperimentalTopK = op::v6::ExperimentalDetectronTopKROIs;

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_topk_rois_eval)
{
    size_t num_rois = 1;

    const auto input_rois_shape = Shape{2, 4};
    const auto input_probs_shape = Shape{2};
    const auto output_shape = Shape{1, 4};

    auto input_rois = std::make_shared<op::Parameter>(element::f32, input_rois_shape);
    auto input_probs = std::make_shared<op::Parameter>(element::f32, input_probs_shape);
    auto topk_rois = std::make_shared<ExperimentalTopK>(input_rois, input_probs, num_rois);
    auto f = std::make_shared<Function>(topk_rois, ParameterVector{input_rois, input_probs});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto topk_rois_output = backend->create_tensor(element::f32, output_shape);

    std::vector<float> input_rois_data = {1.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 5.0f, 7.0f};
    std::vector<float> input_probs_data = {0.5f, 0.3f};
    std::vector<float> expected_result = {1.0, 1.0, 3.0, 4.0};

    auto backend_input_rois_data = backend->create_tensor(element::f32, input_rois_shape);
    copy_data(backend_input_rois_data, input_rois_data);
    auto backend_input_probs_data = backend->create_tensor(element::f32, input_probs_shape);
    copy_data(backend_input_probs_data, input_probs_data);

    auto handle = backend->compile(f);
    handle->call({topk_rois_output}, {backend_input_rois_data, backend_input_probs_data});

    ASSERT_TRUE(test::all_close_f(read_vector<float>(topk_rois_output), expected_result));
}

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_topk_rois_eval_2)
{
    size_t num_rois = 2;

    const auto input_rois_shape = Shape{4, 4};
    const auto input_probs_shape = Shape{4};
    const auto output_shape = Shape{2, 4};

    auto input_rois = std::make_shared<op::Parameter>(element::f32, input_rois_shape);
    auto input_probs = std::make_shared<op::Parameter>(element::f32, input_probs_shape);
    auto topk_rois = std::make_shared<ExperimentalTopK>(input_rois, input_probs, num_rois);
    auto f = std::make_shared<Function>(topk_rois, ParameterVector{input_rois, input_probs});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    auto topk_rois_output = backend->create_tensor(element::f32, output_shape);

    std::vector<float> input_rois_data = {1.0f,  1.0f,  4.0f,  5.0f,  3.0f,  2.0f,  7.0f,  9.0f,
                                          10.0f, 15.0f, 13.0f, 17.0f, 13.0f, 10.0f, 18.0f, 15.0f};
    std::vector<float> input_probs_data = {0.1f, 0.7f, 0.5f, 0.9f};
    std::vector<float> expected_result = {13.0f, 10.0f, 18.0f, 15.0f, 3.0f, 2.0f, 7.0f, 9.0f};

    auto backend_input_rois_data = backend->create_tensor(element::f32, input_rois_shape);
    copy_data(backend_input_rois_data, input_rois_data);
    auto backend_input_probs_data = backend->create_tensor(element::f32, input_probs_shape);
    copy_data(backend_input_probs_data, input_probs_data);

    auto handle = backend->compile(f);
    handle->call({topk_rois_output}, {backend_input_rois_data, backend_input_probs_data});

    ASSERT_TRUE(test::all_close_f(read_vector<float>(topk_rois_output), expected_result));
}
