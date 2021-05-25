// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

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

NGRAPH_TEST(${BACKEND_NAME}, region_yolo_v2_caffe)
{
    const size_t num = 5;
    const size_t coords = 4;
    const size_t classes = 20;
    const size_t batch = 1;
    const size_t channels = 125;
    const size_t width = 13;
    const size_t height = 13;
    const std::vector<int64_t> mask{0, 1, 2};

    Shape input_shape{batch, channels, height, width};
    Shape output_shape{batch, channels * height * width};

    auto A = make_shared<op::Parameter>(element::f32, input_shape);
    auto R = make_shared<op::v0::RegionYolo>(A, coords, classes, num, true, mask, 1, 3);
    auto f = make_shared<Function>(R, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input_from_file<float>(input_shape, TEST_FILES, "region_in_yolov2_caffe.data");
    test_case.add_expected_output_from_file<float>(
        output_shape, TEST_FILES, "region_out_yolov2_caffe.data");
    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, region_yolo_v3_mxnet)
{
    const size_t num = 9;
    const size_t coords = 4;
    const size_t classes = 20;
    const size_t batch = 1;
    const size_t channels = 75;
    const size_t width = 32;
    const size_t height = 32;
    const std::vector<int64_t> mask{0, 1, 2};

    Shape shape{batch, channels, height, width};

    const auto A = make_shared<op::Parameter>(element::f32, shape);
    const auto R = make_shared<op::v0::RegionYolo>(A, coords, classes, num, false, mask, 1, 3);
    const auto f = make_shared<Function>(R, ParameterVector{A});

    EXPECT_EQ(R->get_output_shape(0), shape);

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input_from_file<float>(shape, TEST_FILES, "region_in_yolov3_mxnet.data");
    test_case.add_expected_output_from_file<float>(
        shape, TEST_FILES, "region_out_yolov3_mxnet.data");
    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, region_yolo_v3_mxnet_2)
{
    const size_t num = 1;
    const size_t coords = 4;
    const size_t classes = 1;
    const size_t batch = 1;
    const size_t channels = 8;
    const size_t width = 2;
    const size_t height = 2;
    const std::vector<int64_t> mask{0};
    const int axis = 1;
    const int end_axis = 3;

    Shape input_shape{batch, channels, height, width};
    Shape output_shape{batch, (classes + coords + 1) * mask.size(), height, width};

    const auto A = make_shared<op::Parameter>(element::f32, input_shape);
    const auto R = make_shared<op::v0::RegionYolo>(A, coords, classes, num, false, mask, axis, end_axis);
    const auto f = make_shared<Function>(R, ParameterVector{A});

    EXPECT_EQ(R->get_output_shape(0), output_shape);

    auto test_case = test::TestCase<TestEngine>(f);
    std::vector<float> input {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f
    };

    std::vector<float> output {
        0.52497f, 0.54983f, 0.57444f, 0.59868f, 0.62245f, 0.64565f, 0.66818f, 0.68997f,
        0.1f,     0.2f,     0.3f,     0.4f,     0.5f,     0.6f,     0.7f,     0.8f,
        0.52497f, 0.54983f, 0.57444f, 0.59868f, 0.62245f, 0.64565f, 0.66818f, 0.68997f,
    };

    test_case.add_input<float>(input);
    test_case.add_expected_output<float>(output);
    test_case.run_with_tolerance_as_fp(1.0e-4f);
}
