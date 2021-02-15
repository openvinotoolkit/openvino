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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include <cstring>
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

using Attrs = op::v6::ExperimentalDetectronPriorGridGenerator::Attributes;
using GridGenerator = op::v6::ExperimentalDetectronPriorGridGenerator;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, experimental_detectron_prior_grid_eval)
{
    std::vector<std::vector<float>> priors_value = {
        {-24.5, -12.5, 24.5, 12.5, -16.5, -16.5, 16.5, 16.5, -12.5, -24.5, 12.5, 24.5},
        {-44.5, -24.5, 44.5, 24.5, -32.5, -32.5, 32.5, 32.5, -24.5, -44.5, 24.5, 44.5}};

    struct ShapesAndAttrs
    {
        Attrs attrs;
        Shape priors_shape;
        Shape feature_map_shape;
        Shape im_data_shape;
        Shape ref_out_shape;
    };

    std::vector<ShapesAndAttrs> shapes_and_attrs = {
        {{true, 0, 0, 4.0f, 4.0f}, {3, 4}, {1, 16, 4, 5}, {1, 3, 100, 200}, {60, 4}},
        {{false, 0, 0, 8.0f, 8.0f}, {3, 4}, {1, 16, 3, 7}, {1, 3, 100, 200}, {3, 7, 3, 4}}};

    std::vector<std::vector<float>> expected_results = {
        {-22.5, -10.5, 26.5,  14.5,  -14.5, -14.5, 18.5,  18.5,  -10.5, -22.5, 14.5,  26.5,  -18.5,
         -10.5, 30.5,  14.5,  -10.5, -14.5, 22.5,  18.5,  -6.5,  -22.5, 18.5,  26.5,  -14.5, -10.5,
         34.5,  14.5,  -6.5,  -14.5, 26.5,  18.5,  -2.5,  -22.5, 22.5,  26.5,  -10.5, -10.5, 38.5,
         14.5,  -2.5,  -14.5, 30.5,  18.5,  1.5,   -22.5, 26.5,  26.5,  -6.5,  -10.5, 42.5,  14.5,
         1.5,   -14.5, 34.5,  18.5,  5.5,   -22.5, 30.5,  26.5,  -22.5, -6.5,  26.5,  18.5,  -14.5,
         -10.5, 18.5,  22.5,  -10.5, -18.5, 14.5,  30.5,  -18.5, -6.5,  30.5,  18.5,  -10.5, -10.5,
         22.5,  22.5,  -6.5,  -18.5, 18.5,  30.5,  -14.5, -6.5,  34.5,  18.5,  -6.5,  -10.5, 26.5,
         22.5,  -2.5,  -18.5, 22.5,  30.5,  -10.5, -6.5,  38.5,  18.5,  -2.5,  -10.5, 30.5,  22.5,
         1.5,   -18.5, 26.5,  30.5,  -6.5,  -6.5,  42.5,  18.5,  1.5,   -10.5, 34.5,  22.5,  5.5,
         -18.5, 30.5,  30.5,  -22.5, -2.5,  26.5,  22.5,  -14.5, -6.5,  18.5,  26.5,  -10.5, -14.5,
         14.5,  34.5,  -18.5, -2.5,  30.5,  22.5,  -10.5, -6.5,  22.5,  26.5,  -6.5,  -14.5, 18.5,
         34.5,  -14.5, -2.5,  34.5,  22.5,  -6.5,  -6.5,  26.5,  26.5,  -2.5,  -14.5, 22.5,  34.5,
         -10.5, -2.5,  38.5,  22.5,  -2.5,  -6.5,  30.5,  26.5,  1.5,   -14.5, 26.5,  34.5,  -6.5,
         -2.5,  42.5,  22.5,  1.5,   -6.5,  34.5,  26.5,  5.5,   -14.5, 30.5,  34.5,  -22.5, 1.5,
         26.5,  26.5,  -14.5, -2.5,  18.5,  30.5,  -10.5, -10.5, 14.5,  38.5,  -18.5, 1.5,   30.5,
         26.5,  -10.5, -2.5,  22.5,  30.5,  -6.5,  -10.5, 18.5,  38.5,  -14.5, 1.5,   34.5,  26.5,
         -6.5,  -2.5,  26.5,  30.5,  -2.5,  -10.5, 22.5,  38.5,  -10.5, 1.5,   38.5,  26.5,  -2.5,
         -2.5,  30.5,  30.5,  1.5,   -10.5, 26.5,  38.5,  -6.5,  1.5,   42.5,  26.5,  1.5,   -2.5,
         34.5,  30.5,  5.5,   -10.5, 30.5,  38.5},
        {-40.5, -20.5, 48.5,  28.5,  -28.5, -28.5, 36.5,  36.5,  -20.5, -40.5, 28.5,  48.5,  -32.5,
         -20.5, 56.5,  28.5,  -20.5, -28.5, 44.5,  36.5,  -12.5, -40.5, 36.5,  48.5,  -24.5, -20.5,
         64.5,  28.5,  -12.5, -28.5, 52.5,  36.5,  -4.5,  -40.5, 44.5,  48.5,  -16.5, -20.5, 72.5,
         28.5,  -4.5,  -28.5, 60.5,  36.5,  3.5,   -40.5, 52.5,  48.5,  -8.5,  -20.5, 80.5,  28.5,
         3.5,   -28.5, 68.5,  36.5,  11.5,  -40.5, 60.5,  48.5,  -0.5,  -20.5, 88.5,  28.5,  11.5,
         -28.5, 76.5,  36.5,  19.5,  -40.5, 68.5,  48.5,  7.5,   -20.5, 96.5,  28.5,  19.5,  -28.5,
         84.5,  36.5,  27.5,  -40.5, 76.5,  48.5,  -40.5, -12.5, 48.5,  36.5,  -28.5, -20.5, 36.5,
         44.5,  -20.5, -32.5, 28.5,  56.5,  -32.5, -12.5, 56.5,  36.5,  -20.5, -20.5, 44.5,  44.5,
         -12.5, -32.5, 36.5,  56.5,  -24.5, -12.5, 64.5,  36.5,  -12.5, -20.5, 52.5,  44.5,  -4.5,
         -32.5, 44.5,  56.5,  -16.5, -12.5, 72.5,  36.5,  -4.5,  -20.5, 60.5,  44.5,  3.5,   -32.5,
         52.5,  56.5,  -8.5,  -12.5, 80.5,  36.5,  3.5,   -20.5, 68.5,  44.5,  11.5,  -32.5, 60.5,
         56.5,  -0.5,  -12.5, 88.5,  36.5,  11.5,  -20.5, 76.5,  44.5,  19.5,  -32.5, 68.5,  56.5,
         7.5,   -12.5, 96.5,  36.5,  19.5,  -20.5, 84.5,  44.5,  27.5,  -32.5, 76.5,  56.5,  -40.5,
         -4.5,  48.5,  44.5,  -28.5, -12.5, 36.5,  52.5,  -20.5, -24.5, 28.5,  64.5,  -32.5, -4.5,
         56.5,  44.5,  -20.5, -12.5, 44.5,  52.5,  -12.5, -24.5, 36.5,  64.5,  -24.5, -4.5,  64.5,
         44.5,  -12.5, -12.5, 52.5,  52.5,  -4.5,  -24.5, 44.5,  64.5,  -16.5, -4.5,  72.5,  44.5,
         -4.5,  -12.5, 60.5,  52.5,  3.5,   -24.5, 52.5,  64.5,  -8.5,  -4.5,  80.5,  44.5,  3.5,
         -12.5, 68.5,  52.5,  11.5,  -24.5, 60.5,  64.5,  -0.5,  -4.5,  88.5,  44.5,  11.5,  -12.5,
         76.5,  52.5,  19.5,  -24.5, 68.5,  64.5,  7.5,   -4.5,  96.5,  44.5,  19.5,  -12.5, 84.5,
         52.5,  27.5,  -24.5, 76.5,  64.5}};

    std::size_t i = 0;
    for (const auto& s : shapes_and_attrs)
    {
        auto priors = std::make_shared<op::Parameter>(element::f32, s.priors_shape);
        auto feature_map = std::make_shared<op::Parameter>(element::f32, s.feature_map_shape);
        auto im_data = std::make_shared<op::Parameter>(element::f32, s.im_data_shape);

        auto grid_gen = std::make_shared<GridGenerator>(priors, feature_map, im_data, s.attrs);

        auto f = make_shared<Function>(grid_gen, ParameterVector{priors, feature_map, im_data});

        auto backend = runtime::Backend::create("${BACKEND_NAME}");

        auto priors_data = priors_value[i];

        std::vector<float> ref_results(shape_size(s.ref_out_shape), 0);

        memcpy(ref_results.data(),
               expected_results[i].data(),
               expected_results[i].size() * sizeof(float));

        std::vector<float> feature_map_data(shape_size(s.feature_map_shape));
        std::iota(feature_map_data.begin(), feature_map_data.end(), 0);
        std::vector<float> image_data(shape_size(s.im_data_shape));
        std::iota(image_data.begin(), image_data.end(), 0);

        auto output_priors = backend->create_tensor(element::f32, s.ref_out_shape);

        auto backend_priors = backend->create_tensor(element::f32, s.priors_shape);
        auto backend_feature_map = backend->create_tensor(element::f32, s.feature_map_shape);
        auto backend_im_data = backend->create_tensor(element::f32, s.im_data_shape);
        copy_data(backend_priors, priors_data);
        copy_data(backend_feature_map, feature_map_data);
        copy_data(backend_im_data, image_data);

        auto handle = backend->compile(f);

        handle->call({output_priors}, {backend_priors, backend_feature_map, backend_im_data});

        auto output_priors_value = read_vector<float>(output_priors);

        EXPECT_EQ(ref_results, output_priors_value);
    }
}