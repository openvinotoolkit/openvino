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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "core/null_node.hpp"
#include "gtest/gtest.h"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "default_opset.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_tools.hpp"
#include "util/type_prop.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, onnx_prior_box)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/prior_box.prototxt"));
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    std::vector<float> A(3 * 2 * 2);
    std::vector<float> B(3 * 6 * 6);
    std::vector<float> output = {
        -2.3200002, -2.3200002,  3.6533334,  3.6533334,   -3.7053659,  -3.7053659, 5.0386992,
        5.0386992,  -0.98666668, -2.3200002, 4.9866667,   3.6533334,   -2.3720326, -3.7053659,
        6.3720322,  5.0386992,   -2.3200002, -0.98666668, 3.6533334,   4.9866667,  -3.7053659,
        -2.3720326, 5.0386992,   6.3720322,  -0.98666668, -0.98666668, 4.9866667,  4.9866667,
        -2.3720326, -2.3720326,  6.3720322,  6.3720322,   0.1,         0.1,        0.2,
        0.2,        0.1,         0.1,        0.2,         0.2,         0.1,        0.1,
        0.2,        0.2,         0.1,        0.1,         0.2,         0.2,        0.1,
        0.1,        0.2,         0.2,        0.1,         0.1,         0.2,        0.2,
        0.1,        0.1,         0.2,        0.2,         0.1,         0.1,        0.2,
        0.2,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 32}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/priorbox_clustered.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    std::vector<float> A{15.0};
    std::vector<float> B{10.0};
    std::vector<float> output = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 16}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_most_attrs_default)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/priorbox_clustered_most_attrs_default.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    std::vector<float> A(1 * 1 * 2 * 1);
    std::iota(std::begin(A), std::end(A), 0.0f);
    std::vector<float> B(1 * 1 * 3 * 3);
    std::iota(std::begin(B), std::end(B), 0.0f);
    std::vector<float> output = {-0.1666666716337203979,
                                 -0.1666666716337203979,
                                 0.1666666716337203979,
                                 0.1666666716337203979,
                                 -0.1666666716337203979,
                                 0.3333333432674407959,
                                 0.1666666716337203979,
                                 0.6666666865348815918,
                                 0.1,
                                 0.1,
                                 0.2,
                                 0.2,
                                 0.1,
                                 0.1,
                                 0.2,
                                 0.2};
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 8}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_first_input_bad_shape)
{
    try
    {
        auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/priorbox_clustered_first_input_bad_shape.prototxt"));
        FAIL() << "Expected exception was not thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string("Only 4D inputs are supported. First input rank: 5 (should be 4)"));
    }
    catch (...)
    {
        FAIL() << "Expected OnnxNodeValidationFailure exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_second_input_bad_shape)
{
    try
    {
        auto function = onnx_import::import_onnx_model(file_util::path_join(
            SERIALIZED_ZOO, "onnx/priorbox_clustered_second_input_bad_shape.prototxt"));
        FAIL() << "Expected exception was not thrown";
    }
    catch (const ngraph::ngraph_error& e)
    {
        EXPECT_HAS_SUBSTRING(
            e.what(),
            std::string("Only 4D inputs are supported. Second input rank: 5 (should be 4)"));
    }
    catch (...)
    {
        FAIL() << "Expected OnnxNodeValidationFailure exception was not thrown";
    }
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_detection_output)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/detection_output.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);

    auto gen_vector = [](size_t size, float min, float max) -> std::vector<float> {
        float step = (max - min) / size;
        float next = min - step;

        std::vector<float> out(size);
        std::generate(out.begin(), out.end(), [&next, &step] { return next += step; });
        return out;
    };

    std::vector<float> logits = gen_vector(12, -2, 2);
    std::vector<float> class_preds = gen_vector(9, 0, 1);
    std::vector<float> proposals = gen_vector(12 * 2, 0, 1);
    std::vector<float> output = {0, 1, 0.777778, 0.279849,   0.283779,   0.562743,   0.695387,
                                 0, 1, 0.444444, 0.12963,    0.176075,   0.212963,   0.284573,
                                 0, 2, 0.888889, 0.279849,   0.283779,   0.562743,   0.695387,
                                 0, 2, 0.555556, 0.12963,    0.176075,   0.212963,   0.284573,
                                 0, 2, 0.222222, -0.0608094, -0.0142007, -0.0225239, 0.0304044};
    test_case.add_input<float>(logits);
    test_case.add_input<float>(class_preds);
    test_case.add_input<float>(proposals);
    test_case.add_expected_output<float>(Shape{1, 1, 5, 7}, output);
    int tolerance_bits = 6;
    test_case.run(tolerance_bits);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_group_norm)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/group_norm.prototxt"));
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    Shape shape{2, 8, 2, 2};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_group_norm_5d)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/group_norm_5d.prototxt"));
    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    Shape shape{2, 8, 1, 2, 1};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {-0.34163546562, 0.55278813838, 2.89442372322,  4.68327093124,
                                 -1.02490639686, 1.65836453437, 5.78884744644,  9.36654186248,
                                 -1.70817732810, 2.76394081115, 8.68327140808,  14.04981231689,
                                 -2.39144825935, 3.86951708793, 11.57769489288, 18.73308372497,
                                 -0.34163546562, 0.55278813838, 2.89442372322,  4.68327093124,
                                 -1.02490639686, 1.65836453437, 5.78884744644,  9.36654186248,
                                 -1.70817732810, 2.76394081115, 8.68327140808,  14.04981231689,
                                 -2.39144825935, 3.86951708793, 11.57769489288, 18.73308372497};

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_normalize)
{
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/normalize.prototxt"));
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1);
    std::vector<float> output = {
        0.19334731,
        0.33806169,
        0.44846106,
        0.53452247,
        1.4501048,
        1.5212777,
        1.5696137,
        1.6035674,
        3.4802516,
        3.3806169,
        3.2887144,
        3.2071347,
    };
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 3, 2, 2}, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_swish_with_beta)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/swish_with_beta.prototxt"));

    const Shape expected_output_shape{3};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.2036667, 0.0, 0.2963333});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_swish_without_beta)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO, "onnx/swish_without_beta.prototxt"));

    const Shape expected_output_shape{3};
    auto test_case = test::TestCase<TestEngine>(function);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.18877034, 0.0, 0.31122968});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_detection_output)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/org.openvinotoolkit/experimental_detectron/detection_output.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // rois
    test_case.add_input<float>({1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 4.0f, 1.0f, 8.0f, 5.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{5, 4},
                                         {
                                             0.8929862f,
                                             0.892986297607421875,
                                             12.10701370239257812,
                                             12.10701370239257812,
                                             0,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                             0.0f,
                                         });
    test_case.add_expected_output<int>(Shape{5}, {1, 0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{5}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_detection_output_most_attrs_default)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/"
                             "detection_output_most_attrs_default.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // rois
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{5, 4},
                                         {
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                         });
    test_case.add_expected_output<int>(Shape{5}, {0, 0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_generate_proposals_single_image)
{
    auto function = onnx_import::import_onnx_model(
        file_util::path_join(SERIALIZED_ZOO,
                             "onnx/org.openvinotoolkit/experimental_detectron/"
                             "generate_proposals_single_image.prototxt"));

    auto test_case = test::TestCase<TestEngine>(function);
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});
    // anchors
    test_case.add_input<float>({
        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    });
    // deltas
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
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
         1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(
        Shape{6, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    test_case.add_expected_output<float>(Shape{6}, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_group_norm)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/org.openvinotoolkit/experimental_detectron/group_norm.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);
    Shape shape{2, 8, 2, 2};
    int size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0);
    std::vector<float> output = {
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
        -0.52752507, -0.09108937, 0.3453464, 0.78178215, 2.4364357, 3.309307,  4.1821785, 5.05505,
        -1.5825753,  -0.27326822, 1.0360391, 2.3453465,  4.8728714, 6.618614,  8.364357,  10.1101,
        -2.6376252,  -0.45544672, 1.726732,  3.9089108,  7.309307,  9.927921,  12.546536, 15.165151,
        -3.6926756,  -0.6376257,  2.4174247, 5.472475,   9.745743,  13.237228, 16.728714, 20.2202,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_prior_grid_generator)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/org.openvinotoolkit/experimental_detectron/prior_grid_generator.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);

    std::vector<float> priors(shape_size(Shape{3, 4}));
    std::iota(priors.begin(), priors.end(), 0);

    std::vector<float> feature_map(shape_size(Shape{1, 1, 1, 3}));
    std::iota(feature_map.begin(), feature_map.end(), 0);

    std::vector<float> im_data(shape_size(Shape{1, 3, 4, 7}));
    std::iota(im_data.begin(), im_data.end(), 0);

    test_case.add_input<float>(priors);
    test_case.add_input<float>(feature_map);
    test_case.add_input<float>(im_data);

    test_case.add_expected_output<float>(Shape{9, 4}, {2,  3, 4,  5, 6,  7, 8,  9, 10, 11, 12, 13,
                                                       6,  3, 8,  5, 10, 7, 12, 9, 14, 11, 16, 13,
                                                       10, 3, 12, 5, 14, 7, 16, 9, 18, 11, 20, 13});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_roi_feature_extractor)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO,
        "onnx/org.openvinotoolkit/experimental_detectron/roi_feature_extractor.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);

    std::vector<float> rois(shape_size(Shape{2, 4}));
    std::iota(rois.begin(), rois.end(), 0);

    std::vector<float> pyramid_layer_0(shape_size(Shape{1, 2, 2, 3}));
    std::iota(pyramid_layer_0.begin(), pyramid_layer_0.end(), 0);

    test_case.add_input<float>(rois);
    test_case.add_input<float>(pyramid_layer_0);

    test_case.add_expected_output<float>(Shape{2, 2, 3, 3},
                                         {1.416666746139526367,
                                          1.750000119209289551,
                                          2.083333492279052734,
                                          2.416666746139526367,
                                          2.75,
                                          3.083333492279052734,
                                          3.166666507720947266,
                                          3.5,
                                          3.833333492279052734,
                                          7.416666507720947266,
                                          7.75,
                                          8.083333015441894531,
                                          8.416666984558105469,
                                          8.75,
                                          9.083333969116210938,
                                          9.166666030883789062,
                                          9.5,
                                          9.833333969116210938,
                                          4.166666984558105469,
                                          4.5,
                                          4.833333492279052734,
                                          4.166666984558105469,
                                          4.5,
                                          4.833333492279052734,
                                          2.083333492279052734,
                                          2.25,
                                          2.416666746139526367,
                                          10.16666603088378906,
                                          10.5,
                                          10.83333206176757812,
                                          10.16666603088378906,
                                          10.5,
                                          10.83333206176757812,
                                          5.083333015441894531,
                                          5.25,
                                          5.416666507720947266});

    test_case.add_expected_output<float>(Shape{2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_topk_rios)
{
    auto function = onnx_import::import_onnx_model(file_util::path_join(
        SERIALIZED_ZOO, "onnx/org.openvinotoolkit/experimental_detectron/topk_rios.prototxt"));

    auto test_case = test::TestCase<TestEngine, test::TestCaseType::DYNAMIC>(function);

    test_case.add_input<float>({1.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 5.0f, 7.0f});
    test_case.add_input<float>({0.5f, 0.3f});

    test_case.add_expected_output<float>(Shape{1, 4}, {1, 1, 3, 4});
    test_case.run();
}
