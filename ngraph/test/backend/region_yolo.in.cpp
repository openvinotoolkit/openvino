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

template <typename T>
static bool referenceData2Vector(string file_path, vector<T>& data, int count)
{
    std::ifstream in(file_path, std::ios::binary);
    if (!in.is_open())
    {
        return false;
    }

    T* in_buf = new T[count];
    in.read((char *)in_buf, count * sizeof(T));
    in.close();
    vector<T> in_data(in_buf, in_buf + count);
    data = in_data;
    delete []in_buf;
    return true;
}

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
    const std::vector<float> anchors{10,13,16,30,33,23,30,61,62,45};

    Shape input_shape{batch, channels, height, width};
    Shape output_shape{batch, channels * height * width};

    auto A = make_shared<op::Parameter>(element::f32, input_shape);
    auto R = make_shared<op::v0::RegionYolo>(A, coords, classes, num, true, mask, 1, 3, anchors);
    auto f = make_shared<Function>(R, ParameterVector{A});

    std::vector<float> input;
    std::vector<float> output;

    ASSERT_TRUE(referenceData2Vector<float>("../ngraph/test/files/region_in_yolov2_caffe.data", input, count));
    ASSERT_TRUE(referenceData2Vector<float>("../ngraph/test/files/region_out_yolov2_caffe.data", output, count));

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>(input_shape, input);
    test_case.add_expected_output<float>(output_shape, output);
    test_case.run();
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
    const std::vector<float> anchors{10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326};

    Shape shape{batch, channels, height, width};
    const auto count = shape_size(shape);

    const auto A = make_shared<op::Parameter>(element::f32, shape);
    const auto R = make_shared<op::v0::RegionYolo>(A, coords, classes, num, false, mask, 1, 3, anchors);
    const auto f = make_shared<Function>(R, ParameterVector{A});

    EXPECT_EQ(R->get_output_shape(0), shape);

    std::vector<float> input;
    std::vector<float> output;

    ASSERT_TRUE(referenceData2Vector<float>("../ngraph/test/files/region_in_yolov3_mxnet.data", input, count));
    ASSERT_TRUE(referenceData2Vector<float>("../ngraph/test/files/region_out_yolov3_mxnet.data", output, count));

    auto test_case = test::TestCase<TestEngine>(f);

    test_case.add_input<float>(shape, input);
    test_case.add_expected_output<float>(shape, output);
    test_case.run();
}
