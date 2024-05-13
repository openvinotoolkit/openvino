// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

#include "common_test_utils/all_close.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ndarray.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "common_test_utils/test_tools.hpp"
#include "common_test_utils/type_prop.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");
static std::string s_device = backend_name_to_device("${BACKEND_NAME}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

OPENVINO_TEST(${BACKEND_NAME}, onnx_prior_box) {
    const auto model = convert_model("prior_box.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> A(3 * 2 * 2);
    std::vector<float> B(3 * 6 * 6);
    std::vector<float> output = {
        -2.3200002f,  -2.3200002f,  3.6533334f, 3.6533334f, -3.7053659f, -3.7053659f, 5.0386992f, 5.0386992f,
        -0.98666668f, -2.3200002f,  4.9866667f, 3.6533334f, -2.3720326f, -3.7053659f, 6.3720322f, 5.0386992f,
        -2.3200002f,  -0.98666668f, 3.6533334f, 4.9866667f, -3.7053659f, -2.3720326f, 5.0386992f, 6.3720322f,
        -0.98666668f, -0.98666668f, 4.9866667f, 4.9866667f, -2.3720326f, -2.3720326f, 6.3720322f, 6.3720322f,
        0.1f,         0.1f,         0.2f,       0.2f,       0.1f,        0.1f,        0.2f,       0.2f,
        0.1f,         0.1f,         0.2f,       0.2f,       0.1f,        0.1f,        0.2f,       0.2f,
        0.1f,         0.1f,         0.2f,       0.2f,       0.1f,        0.1f,        0.2f,       0.2f,
        0.1f,         0.1f,         0.2f,       0.2f,       0.1f,        0.1f,        0.2f,       0.2f,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 32}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_priorbox_clustered) {
    auto model = convert_model("priorbox_clustered.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> A{15.0f};
    std::vector<float> B{10.0f};
    std::vector<float> output = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        0.1f, 0.1f, 0.2f, 0.2f, 0.1f, 0.1f, 0.2f, 0.2f, 0.1f, 0.1f, 0.2f, 0.2f, 0.1f, 0.1f, 0.2f, 0.2f,
    };
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 16}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_most_attrs_default) {
    auto model = convert_model("priorbox_clustered_most_attrs_default.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> A(1 * 1 * 2 * 1);
    std::iota(std::begin(A), std::end(A), 0.0f);
    std::vector<float> B(1 * 1 * 3 * 3);
    std::iota(std::begin(B), std::end(B), 0.0f);
    std::vector<float> output = {-0.1666666716337203979f,
                                 -0.1666666716337203979f,
                                 0.1666666716337203979f,
                                 0.1666666716337203979f,
                                 -0.1666666716337203979f,
                                 0.3333333432674407959f,
                                 0.1666666716337203979f,
                                 0.6666666865348815918f,
                                 0.1f,
                                 0.1f,
                                 0.2f,
                                 0.2f,
                                 0.1f,
                                 0.1f,
                                 0.2f,
                                 0.2f};
    test_case.add_input<float>(A);
    test_case.add_input<float>(B);
    test_case.add_expected_output<float>(Shape{1, 2, 8}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_first_input_bad_shape) {
    try {
        auto model = convert_model("priorbox_clustered_first_input_bad_shape.onnx");
        FAIL() << "Expected exception was not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Only 4D inputs are supported. First input rank: 5 (should be 4)"));
    } catch (...) {
        FAIL() << "Expected ov::Exception exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_priorbox_clustered_second_input_bad_shape) {
    try {
        auto model = convert_model("priorbox_clustered_second_input_bad_shape.onnx");
        FAIL() << "Expected exception was not thrown";
    } catch (const ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Only 4D inputs are supported. Second input rank: 5 (should be 4)"));
    } catch (...) {
        FAIL() << "Expected ov::Exception exception was not thrown";
    }
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_detection_output) {
    const auto model = convert_model("detection_output.onnx");
    auto test_case = ov::test::TestCase(model, s_device);

    auto gen_vector = [](size_t size, float min, float max) -> std::vector<float> {
        float step = (max - min) / size;
        float next = min - step;

        std::vector<float> out(size);
        std::generate(out.begin(), out.end(), [&next, &step] {
            return next += step;
        });
        return out;
    };

    std::vector<float> logits = gen_vector(12, -2, 2);
    std::vector<float> class_preds = gen_vector(9, 0, 1);
    std::vector<float> proposals = gen_vector(12 * 2, 0, 1);
    std::vector<float> output = {0, 1, 0.777778f, 0.279849f,   0.283779f,   0.562743f,   0.695387f,
                                 0, 1, 0.444444f, 0.12963f,    0.176075f,   0.212963f,   0.284573f,
                                 0, 2, 0.888889f, 0.279849f,   0.283779f,   0.562743f,   0.695387f,
                                 0, 2, 0.555556f, 0.12963f,    0.176075f,   0.212963f,   0.284573f,
                                 0, 2, 0.222222f, -0.0608094f, -0.0142007f, -0.0225239f, 0.0304044f};
    test_case.add_input<float>(logits);
    test_case.add_input<float>(class_preds);
    test_case.add_input<float>(proposals);
    test_case.add_expected_output<float>(Shape{1, 1, 5, 7}, output);
    int tolerance_bits = 6;
    test_case.run(tolerance_bits);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_group_norm) {
    const auto model = convert_model("group_norm.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    Shape shape{2, 8, 2, 2};
    const auto size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.f);
    std::vector<float> output = {
        -0.52752507f, -0.09108937f, 0.3453464f, 0.78178215f, 2.4364357f, 3.309307f,  4.1821785f, 5.05505f,
        -1.5825753f,  -0.27326822f, 1.0360391f, 2.3453465f,  4.8728714f, 6.618614f,  8.364357f,  10.1101f,
        -2.6376252f,  -0.45544672f, 1.726732f,  3.9089108f,  7.309307f,  9.927921f,  12.546536f, 15.165151f,
        -3.6926756f,  -0.6376257f,  2.4174247f, 5.472475f,   9.745743f,  13.237228f, 16.728714f, 20.2202f,
        -0.52752507f, -0.09108937f, 0.3453464f, 0.78178215f, 2.4364357f, 3.309307f,  4.1821785f, 5.05505f,
        -1.5825753f,  -0.27326822f, 1.0360391f, 2.3453465f,  4.8728714f, 6.618614f,  8.364357f,  10.1101f,
        -2.6376252f,  -0.45544672f, 1.726732f,  3.9089108f,  7.309307f,  9.927921f,  12.546536f, 15.165151f,
        -3.6926756f,  -0.6376257f,  2.4174247f, 5.472475f,   9.745743f,  13.237228f, 16.728714f, 20.2202f,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_group_norm_squeeze_bias_and_scale) {
    const auto model = convert_model("group_norm_4D_bias_and_scale.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    Shape shape{2, 8, 2, 2};
    const auto size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.f);
    std::vector<float> output = {
        -0.52752507f, -0.09108937f, 0.3453464f, 0.78178215f, 2.4364357f, 3.309307f,  4.1821785f, 5.05505f,
        -1.5825753f,  -0.27326822f, 1.0360391f, 2.3453465f,  4.8728714f, 6.618614f,  8.364357f,  10.1101f,
        -2.6376252f,  -0.45544672f, 1.726732f,  3.9089108f,  7.309307f,  9.927921f,  12.546536f, 15.165151f,
        -3.6926756f,  -0.6376257f,  2.4174247f, 5.472475f,   9.745743f,  13.237228f, 16.728714f, 20.2202f,
        -0.52752507f, -0.09108937f, 0.3453464f, 0.78178215f, 2.4364357f, 3.309307f,  4.1821785f, 5.05505f,
        -1.5825753f,  -0.27326822f, 1.0360391f, 2.3453465f,  4.8728714f, 6.618614f,  8.364357f,  10.1101f,
        -2.6376252f,  -0.45544672f, 1.726732f,  3.9089108f,  7.309307f,  9.927921f,  12.546536f, 15.165151f,
        -3.6926756f,  -0.6376257f,  2.4174247f, 5.472475f,   9.745743f,  13.237228f, 16.728714f, 20.2202f,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_group_norm_5d) {
    const auto model = convert_model("group_norm_5d.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    Shape shape{2, 8, 1, 2, 1};
    const auto size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.f);
    std::vector<float> output = {-0.34163546562f, 0.55278813838f,  2.89442372322f,  4.68327093124f,  -1.02490639686f,
                                 1.65836453437f,  5.78884744644f,  9.36654186248f,  -1.70817732810f, 2.76394081115f,
                                 8.68327140808f,  14.04981231689f, -2.39144825935f, 3.86951708793f,  11.57769489288f,
                                 18.73308372497f, -0.34163546562f, 0.55278813838f,  2.89442372322f,  4.68327093124f,
                                 -1.02490639686f, 1.65836453437f,  5.78884744644f,  9.36654186248f,  -1.70817732810f,
                                 2.76394081115f,  8.68327140808f,  14.04981231689f, -2.39144825935f, 3.86951708793f,
                                 11.57769489288f, 18.73308372497f};

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_normalize) {
    const auto model = convert_model("normalize.onnx");
    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 1.f);
    std::vector<float> output = {
        0.19334731f,
        0.33806169f,
        0.44846106f,
        0.53452247f,
        1.4501048f,
        1.5212777f,
        1.5696137f,
        1.6035674f,
        3.4802516f,
        3.3806169f,
        3.2887144f,
        3.2071347f,
    };
    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(Shape{1, 3, 2, 2}, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_swish_with_beta) {
    auto model = convert_model("swish_with_beta.onnx");

    const Shape expected_output_shape{3};
    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.2036667f, 0.0f, 0.2963333f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_swish_without_beta) {
    auto model = convert_model("swish_without_beta.onnx");

    const Shape expected_output_shape{3};
    auto test_case = ov::test::TestCase(model, s_device);
    std::vector<float> input_data{-0.5f, 0, 0.5f};
    test_case.add_input<float>(input_data);
    test_case.add_expected_output<float>(expected_output_shape, {-0.18877034f, 0.0f, 0.31122968f});

    test_case.run_with_tolerance_as_fp(2.0e-5f);
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_detection_output) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/detection_output.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // rois
    test_case.add_input<float>({1.0f, 1.0f, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  4.0f,  1.0f, 8.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f,  1.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{5, 4},
                                         {
                                             0.8929862f,
                                             0.892986297607421875f,
                                             12.10701370239257812f,
                                             12.10701370239257812f,
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

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_detection_output_most_attrs_default) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/"
                               "detection_output_most_attrs_default.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // rois
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 8.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});

    test_case.add_expected_output<float>(Shape{5, 4},
                                         {
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                             0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                         });
    test_case.add_expected_output<int>(Shape{5}, {0, 0, 0, 0, 0});
    test_case.add_expected_output<float>(Shape{5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_generate_proposals_single_image) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/"
                               "generate_proposals_single_image.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    // im_info
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});
    // anchors
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // deltas
    test_case.add_input<float>(
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,

         1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    // scores
    test_case.add_input<float>({
        5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 8.0f, 1.0f,
    });

    test_case.add_expected_output<float>(Shape{6, 4},
                                         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    test_case.add_expected_output<float>(Shape{6}, {8.0f, 5.0f, 4.0f, 1.0f, 1.0f, 1.0f});

    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_group_norm) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/group_norm.onnx");

    auto test_case = ov::test::TestCase(model, s_device);
    Shape shape{2, 8, 2, 2};
    const auto size = shape_size(shape);
    std::vector<float> data(size);
    std::iota(data.begin(), data.end(), 0.f);
    std::vector<float> output = {
        -0.52752507f, -0.09108937f, 0.3453464f, 0.78178215f, 2.4364357f, 3.309307f,  4.1821785f, 5.05505f,
        -1.5825753f,  -0.27326822f, 1.0360391f, 2.3453465f,  4.8728714f, 6.618614f,  8.364357f,  10.1101f,
        -2.6376252f,  -0.45544672f, 1.726732f,  3.9089108f,  7.309307f,  9.927921f,  12.546536f, 15.165151f,
        -3.6926756f,  -0.6376257f,  2.4174247f, 5.472475f,   9.745743f,  13.237228f, 16.728714f, 20.2202f,
        -0.52752507f, -0.09108937f, 0.3453464f, 0.78178215f, 2.4364357f, 3.309307f,  4.1821785f, 5.05505f,
        -1.5825753f,  -0.27326822f, 1.0360391f, 2.3453465f,  4.8728714f, 6.618614f,  8.364357f,  10.1101f,
        -2.6376252f,  -0.45544672f, 1.726732f,  3.9089108f,  7.309307f,  9.927921f,  12.546536f, 15.165151f,
        -3.6926756f,  -0.6376257f,  2.4174247f, 5.472475f,   9.745743f,  13.237228f, 16.728714f, 20.2202f,
    };

    test_case.add_input<float>(data);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_prior_grid_generator) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/prior_grid_generator.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    std::vector<float> priors(shape_size(Shape{3, 4}));
    std::iota(priors.begin(), priors.end(), 0.f);

    std::vector<float> feature_map(shape_size(Shape{1, 1, 1, 3}));
    std::iota(feature_map.begin(), feature_map.end(), 0.f);

    std::vector<float> im_data(shape_size(Shape{1, 3, 4, 7}));
    std::iota(im_data.begin(), im_data.end(), 0.f);

    test_case.add_input<float>(priors);
    test_case.add_input<float>(feature_map);
    test_case.add_input<float>(im_data);

    test_case.add_expected_output<float>(Shape{9, 4},
                                         {2,  3, 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 6,  3, 8,  5,  10, 7,
                                          12, 9, 14, 11, 16, 13, 10, 3, 12, 5,  14, 7,  16, 9, 18, 11, 20, 13});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_roi_feature_extractor) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/roi_feature_extractor.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    std::vector<float> rois(shape_size(Shape{2, 4}));
    std::iota(rois.begin(), rois.end(), 0.f);

    std::vector<float> pyramid_layer_0(shape_size(Shape{1, 2, 2, 3}));
    std::iota(pyramid_layer_0.begin(), pyramid_layer_0.end(), 0.f);

    test_case.add_input<float>(rois);
    test_case.add_input<float>(pyramid_layer_0);

    test_case.add_expected_output<float>(Shape{2, 2, 3, 3},
                                         {1.416666746139526367f,
                                          1.750000119209289551f,
                                          2.083333492279052734f,
                                          2.416666746139526367f,
                                          2.75f,
                                          3.083333492279052734f,
                                          3.166666507720947266f,
                                          3.5f,
                                          3.833333492279052734f,
                                          7.416666507720947266f,
                                          7.75f,
                                          8.083333015441894531f,
                                          8.416666984558105469f,
                                          8.75f,
                                          9.083333969116210938f,
                                          9.166666030883789062f,
                                          9.5f,
                                          9.833333969116210938f,
                                          4.166666984558105469f,
                                          4.5f,
                                          4.833333492279052734f,
                                          4.166666984558105469f,
                                          4.5f,
                                          4.833333492279052734f,
                                          2.083333492279052734f,
                                          2.25f,
                                          2.416666746139526367f,
                                          10.16666603088378906f,
                                          10.5f,
                                          10.83333206176757812f,
                                          10.16666603088378906f,
                                          10.5f,
                                          10.83333206176757812f,
                                          5.083333015441894531f,
                                          5.25f,
                                          5.416666507720947266f});

    test_case.add_expected_output<float>(Shape{2, 4}, {0, 1, 2, 3, 4, 5, 6, 7});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_experimental_detectron_topk_rios) {
    auto model = convert_model("org.openvinotoolkit/experimental_detectron/topk_rios.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    test_case.add_input<float>({1.0f, 1.0f, 3.0f, 4.0f, 2.0f, 1.0f, 5.0f, 7.0f});
    test_case.add_input<float>({0.5f, 0.3f});

    test_case.add_expected_output<float>(Shape{1, 4}, {1, 1, 3, 4});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_deformable_conv_2d) {
    auto model = convert_model("org.openvinotoolkit/deformable_conv_2d.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // data
    test_case.add_input<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    // deformations
    test_case.add_input<float>({0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,
                                1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f,
                                0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,
                                0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,
                                1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f,
                                0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f});

    test_case.add_expected_output<float>(Shape{1, 1, 3, 3},
                                         {6.9000001f,
                                          2.8500001f,
                                          6.4000001f,
                                          13.4000006f,
                                          11.8999996f,
                                          7.9000006f,
                                          12.4000006f,
                                          4.6999998f,
                                          1.6000000f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_deformable_conv_2d_with_mask) {
    auto model = convert_model("org.openvinotoolkit/deformable_conv_2d_with_mask.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // data
    test_case.add_input<float>(
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    // deformations
    test_case.add_input<float>({0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,
                                1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f,
                                0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,
                                0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,
                                1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f,
                                0.0f, 1.0f,  1.0f,  0.5f, -0.5f, 0.0f,  1.0f, 0.5f,  -0.5f, 0.0f, 1.0f,  1.0f});

    // mask
    test_case.add_input<float>({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
                                1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
                                2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f});

    test_case.add_expected_output<float>(Shape{1, 1, 3, 3},
                                         {14.7299995f,
                                          7.3200006f,
                                          15.0600004f,
                                          31.1000004f,
                                          28.9899998f,
                                          20.5800018f,
                                          32.6200027f,
                                          6.6400003f,
                                          1.4399999f});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_generate_proposals) {
    auto model = convert_model("org.openvinotoolkit/generate_proposals.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // scores
    test_case.add_input<float>(
        Shape{1, 3, 2, 6},
        {0.56637216f, 0.90457034f, 0.69827306f, 0.4353543f,  0.47985056f, 0.42658508f, 0.14516132f, 0.08081771f,
         0.1799732f,  0.9229515f,  0.42420176f, 0.50857586f, 0.82664067f, 0.4972319f,  0.3752427f,  0.56731623f,
         0.18241242f, 0.33252355f, 0.30608943f, 0.6572437f,  0.69185436f, 0.88646156f, 0.36985755f, 0.5590753f,
         0.5256446f,  0.03342898f, 0.1344396f,  0.68642473f, 0.37953874f, 0.32575172f, 0.21108444f, 0.5661886f,
         0.45378175f, 0.62126315f, 0.26799858f, 0.37272978f});
    // deltas
    test_case.add_input<float>(
        Shape{1, 12, 2, 6},
        {0.5337073f,  0.86607957f, 0.55151343f, 0.21626699f, 0.4462629f,  0.03985678f, 0.5157072f,  0.9932138f,
         0.7565954f,  0.43803605f, 0.802818f,   0.14834064f, 0.53932905f, 0.14314f,    0.3817048f,  0.95075196f,
         0.05516243f, 0.2567484f,  0.25508744f, 0.77438325f, 0.43561f,    0.2094628f,  0.8299043f,  0.44982538f,
         0.95615596f, 0.5651084f,  0.11801951f, 0.05352486f, 0.9774733f,  0.14439464f, 0.62644225f, 0.14370479f,
         0.54161614f, 0.557915f,   0.53102225f, 0.0840179f,  0.7249888f,  0.9843559f,  0.5490522f,  0.53788143f,
         0.822474f,   0.3278008f,  0.39688024f, 0.3286012f,  0.5117038f,  0.04743988f, 0.9408995f,  0.29885054f,
         0.81039643f, 0.85277915f, 0.06807619f, 0.86430097f, 0.36225632f, 0.16606331f, 0.5401001f,  0.7541649f,
         0.11998601f, 0.5131829f,  0.40606487f, 0.327888f,   0.27721855f, 0.6378373f,  0.22795396f, 0.4961256f,
         0.3215895f,  0.15607187f, 0.14782153f, 0.8908137f,  0.8835288f,  0.834191f,   0.29907143f, 0.7983525f,
         0.755875f,   0.30837986f, 0.0839176f,  0.26624718f, 0.04371626f, 0.09472824f, 0.20689541f, 0.37622106f,
         0.1083321f,  0.1342548f,  0.05815459f, 0.7676379f,  0.8105144f,  0.92348766f, 0.26761323f, 0.7183306f,
         0.8947588f,  0.19020908f, 0.42731014f, 0.7473663f,  0.85775334f, 0.9340091f,  0.3278848f,  0.755993f,
         0.05307213f, 0.39705503f, 0.21003333f, 0.5625373f,  0.66188884f, 0.80521655f, 0.6125863f,  0.44678232f,
         0.97802377f, 0.0204936f,  0.02686367f, 0.7390654f,  0.74631f,    0.58399844f, 0.5988792f,  0.37413648f,
         0.5946692f,  0.6955776f,  0.36377597f, 0.7891322f,  0.40900692f, 0.99139464f, 0.50169915f, 0.41435778f,
         0.17142445f, 0.26761186f, 0.31591868f, 0.14249913f, 0.12919712f, 0.5418711f,  0.6523203f,  0.50259084f,
         0.7379765f,  0.01171071f, 0.94423133f, 0.00841132f, 0.97486794f, 0.2921785f,  0.7633071f,  0.88477814f,
         0.03563205f, 0.50833166f, 0.01354555f, 0.535081f,   0.41366324f, 0.0694767f,  0.9944055f,  0.9981207f});
    // im_info
    test_case.add_input<float>(Shape{1, 3}, {200, 200, 0});
    // anchors
    test_case.add_input<float>(Shape{3, 4}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<float>(
        Shape{6, 4},
        {0.12904608f, 1.3703424f, 3.6230984f, 3.4675088f, 0.9725206f, 0.,         4.4917974f, 4.9623675f,
         4.882682f,   5.1236916f, 7.1700497f, 10.213073f, 4.4913187f, 4.305372f,  8.750267f,  8.803502f,
         0.9777608f,  1.0317986f, 3.228293f,  4.495021f,  4.125554f,  5.4091997f, 6.35439f,   10.124915f});
    test_case.add_expected_output<float>(Shape{6},
                                         {0.9229515f, 0.90457034f, 0.88646156f, 0.82664067f, 0.69827306f, 0.69185436f});
    test_case.add_expected_output<int64_t>(Shape{1}, {6});
    test_case.run();
}

OPENVINO_TEST(${BACKEND_NAME}, onnx_model_generate_proposals_batch) {
    auto model = convert_model("org.openvinotoolkit/generate_proposals_batch2.onnx");

    auto test_case = ov::test::TestCase(model, s_device);

    // scores
    test_case.add_input<float>(Shape{2, 3, 2, 3}, {5, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 7, 1, 1, 1, 1,
                                                   1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 8, 1});
    // deltas
    test_case.add_input<float>(Shape{2, 12, 2, 3}, std::vector<float>(144, 1));
    // im_info
    test_case.add_input<float>(Shape{2, 3}, {1, 1, 0, 1, 1, 0});
    // anchors
    test_case.add_input<float>(Shape{3, 4}, std::vector<float>(12, 1));

    test_case.add_expected_output<float>(Shape{10, 4}, std::vector<float>(40, 1));
    test_case.add_expected_output<float>(Shape{10}, {7, 5, 3, 1, 1, 8, 4, 2, 1, 1});
    test_case.add_expected_output<int64_t>(Shape{2}, {5, 5});
    test_case.run();
}
