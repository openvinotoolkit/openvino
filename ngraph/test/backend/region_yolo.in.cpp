//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
    const size_t count = width * height * channels;
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
    const auto count = shape_size(shape);

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
