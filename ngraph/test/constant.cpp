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

#include <memory>

#include <gtest/gtest.h>

#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace std;

//
// boolean
//

TEST(constant, boolean_string)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, boolean_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<string>{"1"});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, boolean_vector)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<char>{1, 0, 1, 0});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, boolean_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::boolean, shape, vector<char>{1});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const char* p = c.get_data_ptr<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// float
//

TEST(constant, float_string)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, float_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<string>{"1"});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, float_vector)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<float>{1, 0, 1, 0});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, float_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f32, shape, vector<float>{1});
    auto v = c.get_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const float* p = c.get_data_ptr<float>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// double
//

TEST(constant, double_string)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, double_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<string>{"1"});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, double_vector)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<double>{1, 0, 1, 0});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, double_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f64, shape, vector<double>{1});
    auto v = c.get_vector<double>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const double* p = c.get_data_ptr<double>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int8
//

TEST(constant, int8_string)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int8_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<string>{"1"});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int8_vector)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<int8_t>{1, 0, 1, 0});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int8_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i8, shape, vector<int8_t>{1});
    auto v = c.get_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int8_t* p = c.get_data_ptr<int8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int16
//

TEST(constant, int16_string)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<string>{"1"});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int16_vector)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<int16_t>{1, 0, 1, 0});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i16, shape, vector<int16_t>{1});
    auto v = c.get_vector<int16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int16_t* p = c.get_data_ptr<int16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int32
//

TEST(constant, int32_string)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int32_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<string>{"1"});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int32_vector)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<int32_t>{1, 0, 1, 0});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int32_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i32, shape, vector<int32_t>{1});
    auto v = c.get_vector<int32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int32_t* p = c.get_data_ptr<int32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// int64
//

TEST(constant, int64_string)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int64_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<string>{"1"});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, int64_vector)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<int64_t>{1, 0, 1, 0});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, int64_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::i64, shape, vector<int64_t>{1});
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const int64_t* p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint8
//

TEST(constant, uint8_string)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint8_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<string>{"1"});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint8_vector)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<uint8_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint8_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u8, shape, vector<uint8_t>{1});
    auto v = c.get_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint8_t* p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint16
//

TEST(constant, uint16_string)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<string>{"1"});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint16_vector)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<uint16_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u16, shape, vector<uint16_t>{1});
    auto v = c.get_vector<uint16_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint16_t* p = c.get_data_ptr<uint16_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint32
//

TEST(constant, uint32_string)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint32_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<string>{"1"});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint32_vector)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<uint32_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint32_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u32, shape, vector<uint32_t>{1});
    auto v = c.get_vector<uint32_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint32_t* p = c.get_data_ptr<uint32_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// uint64
//

TEST(constant, uint64_string)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint64_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<string>{"1"});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

TEST(constant, uint64_vector)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<uint64_t>{1, 0, 1, 0});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, uint64_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::u64, shape, vector<uint64_t>{1});
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const uint64_t* p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 1);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 1);
}

//
// bfloat16
//

TEST(constant, bfloat16_string)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(0));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(0));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(0));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(0));
}

TEST(constant, bfloat16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<string>{"1"});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(1));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(1));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(1));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(1));
}

TEST(constant, bfloat16_vector)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<bfloat16>{1, 0, 1, 0});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(0));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(0));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(0));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(0));
}

TEST(constant, bfloat16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::bf16, shape, vector<bfloat16>{1});
    auto v = c.get_vector<bfloat16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], bfloat16(1));
    EXPECT_EQ(v[1], bfloat16(1));
    EXPECT_EQ(v[2], bfloat16(1));
    EXPECT_EQ(v[3], bfloat16(1));

    const bfloat16* p = c.get_data_ptr<bfloat16>();
    EXPECT_EQ(p[0], bfloat16(1));
    EXPECT_EQ(p[1], bfloat16(1));
    EXPECT_EQ(p[2], bfloat16(1));
    EXPECT_EQ(p[3], bfloat16(1));
}

//
// float16
//

TEST(constant, float16_string)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<string>{"1", "0", "1", "0"});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(0));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(0));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(0));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(0));
}

TEST(constant, float16_string_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<string>{"1"});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(1));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(1));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(1));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(1));
}

TEST(constant, float16_vector)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<float16>{1, 0, 1, 0});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(0));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(0));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(0));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(0));
}

TEST(constant, float16_vector_broadcast)
{
    Shape shape{4};
    op::Constant c(element::f16, shape, vector<float16>{1});
    auto v = c.get_vector<float16>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], float16(1));
    EXPECT_EQ(v[1], float16(1));
    EXPECT_EQ(v[2], float16(1));
    EXPECT_EQ(v[3], float16(1));

    const float16* p = c.get_data_ptr<float16>();
    EXPECT_EQ(p[0], float16(1));
    EXPECT_EQ(p[1], float16(1));
    EXPECT_EQ(p[2], float16(1));
    EXPECT_EQ(p[3], float16(1));
}

TEST(constant, shared_data)
{
    Shape shape{100, 200};
    auto c1 = make_shared<op::Constant>(element::f16, shape, vector<float16>{123});
    auto c2 = static_pointer_cast<op::Constant>(c1->clone_with_new_inputs({}));
    const float* p1 = c1->get_data_ptr<float>();
    const float* p2 = c2->get_data_ptr<float>();
    EXPECT_EQ(p1, p2);
}

template <typename T1, typename T2>
::testing::AssertionResult test_convert()
{
    Shape shape{5};
    vector<T1> expected{1, 2, 3, 4, 5};
    auto c1 = make_shared<op::Constant>(element::from<T2>(), shape, expected);
    vector<T1> actual = c1->template cast_vector<T1>();
    ::testing::AssertionResult rc =
        (actual == expected ? ::testing::AssertionSuccess() : ::testing::AssertionFailure());
    rc << "Conversion failed";
    return rc;
}

TEST(constant, convert_input)
{
    EXPECT_TRUE((test_convert<float, float>()));
    EXPECT_TRUE((test_convert<float, double>()));
    EXPECT_TRUE((test_convert<float, float16>()));
    EXPECT_TRUE((test_convert<float, bfloat16>()));
    EXPECT_TRUE((test_convert<float, int8_t>()));
    EXPECT_TRUE((test_convert<float, int16_t>()));
    EXPECT_TRUE((test_convert<float, int32_t>()));
    EXPECT_TRUE((test_convert<float, int64_t>()));
    EXPECT_TRUE((test_convert<float, uint8_t>()));
    EXPECT_TRUE((test_convert<float, uint16_t>()));
    EXPECT_TRUE((test_convert<float, uint32_t>()));
    EXPECT_TRUE((test_convert<float, uint64_t>()));

    EXPECT_TRUE((test_convert<double, float>()));
    EXPECT_TRUE((test_convert<double, double>()));
    EXPECT_TRUE((test_convert<double, float16>()));
    EXPECT_TRUE((test_convert<double, bfloat16>()));
    EXPECT_TRUE((test_convert<double, int8_t>()));
    EXPECT_TRUE((test_convert<double, int16_t>()));
    EXPECT_TRUE((test_convert<double, int32_t>()));
    EXPECT_TRUE((test_convert<double, int64_t>()));
    EXPECT_TRUE((test_convert<double, uint8_t>()));
    EXPECT_TRUE((test_convert<double, uint16_t>()));
    EXPECT_TRUE((test_convert<double, uint32_t>()));
    EXPECT_TRUE((test_convert<double, uint64_t>()));

    EXPECT_TRUE((test_convert<float16, float>()));
    EXPECT_TRUE((test_convert<float16, double>()));
    EXPECT_TRUE((test_convert<float16, float16>()));
    EXPECT_TRUE((test_convert<float16, bfloat16>()));
    EXPECT_TRUE((test_convert<float16, int8_t>()));
    EXPECT_TRUE((test_convert<float16, int16_t>()));
    EXPECT_TRUE((test_convert<float16, int32_t>()));
    EXPECT_TRUE((test_convert<float16, int64_t>()));
    EXPECT_TRUE((test_convert<float16, uint8_t>()));
    EXPECT_TRUE((test_convert<float16, uint16_t>()));
    EXPECT_TRUE((test_convert<float16, uint32_t>()));
    EXPECT_TRUE((test_convert<float16, uint64_t>()));

    EXPECT_TRUE((test_convert<bfloat16, float>()));
    EXPECT_TRUE((test_convert<bfloat16, double>()));
    EXPECT_TRUE((test_convert<bfloat16, float16>()));
    EXPECT_TRUE((test_convert<bfloat16, bfloat16>()));
    EXPECT_TRUE((test_convert<bfloat16, int8_t>()));
    EXPECT_TRUE((test_convert<bfloat16, int16_t>()));
    EXPECT_TRUE((test_convert<bfloat16, int32_t>()));
    EXPECT_TRUE((test_convert<bfloat16, int64_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint8_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint16_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint32_t>()));
    EXPECT_TRUE((test_convert<bfloat16, uint64_t>()));

    EXPECT_TRUE((test_convert<int8_t, float>()));
    EXPECT_TRUE((test_convert<int8_t, double>()));
    EXPECT_TRUE((test_convert<int8_t, float16>()));
    EXPECT_TRUE((test_convert<int8_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int8_t, int8_t>()));
    EXPECT_TRUE((test_convert<int8_t, int16_t>()));
    EXPECT_TRUE((test_convert<int8_t, int32_t>()));
    EXPECT_TRUE((test_convert<int8_t, int64_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int8_t, uint64_t>()));

    EXPECT_TRUE((test_convert<int16_t, float>()));
    EXPECT_TRUE((test_convert<int16_t, double>()));
    EXPECT_TRUE((test_convert<int16_t, float16>()));
    EXPECT_TRUE((test_convert<int16_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int16_t, int8_t>()));
    EXPECT_TRUE((test_convert<int16_t, int16_t>()));
    EXPECT_TRUE((test_convert<int16_t, int32_t>()));
    EXPECT_TRUE((test_convert<int16_t, int64_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int16_t, uint64_t>()));

    EXPECT_TRUE((test_convert<int32_t, float>()));
    EXPECT_TRUE((test_convert<int32_t, double>()));
    EXPECT_TRUE((test_convert<int32_t, float16>()));
    EXPECT_TRUE((test_convert<int32_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int32_t, int8_t>()));
    EXPECT_TRUE((test_convert<int32_t, int16_t>()));
    EXPECT_TRUE((test_convert<int32_t, int32_t>()));
    EXPECT_TRUE((test_convert<int32_t, int64_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int32_t, uint64_t>()));

    EXPECT_TRUE((test_convert<int64_t, float>()));
    EXPECT_TRUE((test_convert<int64_t, double>()));
    EXPECT_TRUE((test_convert<int64_t, float16>()));
    EXPECT_TRUE((test_convert<int64_t, bfloat16>()));
    EXPECT_TRUE((test_convert<int64_t, int8_t>()));
    EXPECT_TRUE((test_convert<int64_t, int16_t>()));
    EXPECT_TRUE((test_convert<int64_t, int32_t>()));
    EXPECT_TRUE((test_convert<int64_t, int64_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint8_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint16_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint32_t>()));
    EXPECT_TRUE((test_convert<int64_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint8_t, float>()));
    EXPECT_TRUE((test_convert<uint8_t, double>()));
    EXPECT_TRUE((test_convert<uint8_t, float16>()));
    EXPECT_TRUE((test_convert<uint8_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint8_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint8_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint8_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint8_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint8_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint16_t, float>()));
    EXPECT_TRUE((test_convert<uint16_t, double>()));
    EXPECT_TRUE((test_convert<uint16_t, float16>()));
    EXPECT_TRUE((test_convert<uint16_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint16_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint16_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint16_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint16_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint16_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint32_t, float>()));
    EXPECT_TRUE((test_convert<uint32_t, double>()));
    EXPECT_TRUE((test_convert<uint32_t, float16>()));
    EXPECT_TRUE((test_convert<uint32_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint32_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint32_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint32_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint32_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint32_t, uint64_t>()));

    EXPECT_TRUE((test_convert<uint64_t, float>()));
    EXPECT_TRUE((test_convert<uint64_t, double>()));
    EXPECT_TRUE((test_convert<uint64_t, float16>()));
    EXPECT_TRUE((test_convert<uint64_t, bfloat16>()));
    EXPECT_TRUE((test_convert<uint64_t, int8_t>()));
    EXPECT_TRUE((test_convert<uint64_t, int16_t>()));
    EXPECT_TRUE((test_convert<uint64_t, int32_t>()));
    EXPECT_TRUE((test_convert<uint64_t, int64_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint8_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint16_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint32_t>()));
    EXPECT_TRUE((test_convert<uint64_t, uint64_t>()));
}

template <typename T1, typename T2>
::testing::AssertionResult test_uniform_ctor()
{
    Shape shape{5};
    vector<T1> expected{3, 3, 3, 3, 3};
    auto c1 = make_shared<op::Constant>(element::from<T2>(), shape, 3);
    vector<T1> actual = c1->template cast_vector<T1>();
    ::testing::AssertionResult rc =
        (actual == expected ? ::testing::AssertionSuccess() : ::testing::AssertionFailure());
    rc << "Construction of uniform Constant failed";
    return rc;
}

TEST(constant, construct_uniform)
{
    EXPECT_TRUE((test_uniform_ctor<float, float>()));
    EXPECT_TRUE((test_uniform_ctor<float, double>()));
    EXPECT_TRUE((test_uniform_ctor<float, float16>()));
    EXPECT_TRUE((test_uniform_ctor<float, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<float, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<double, float>()));
    EXPECT_TRUE((test_uniform_ctor<double, double>()));
    EXPECT_TRUE((test_uniform_ctor<double, float16>()));
    EXPECT_TRUE((test_uniform_ctor<double, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<double, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<double, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<float16, float>()));
    EXPECT_TRUE((test_uniform_ctor<float16, double>()));
    EXPECT_TRUE((test_uniform_ctor<float16, float16>()));
    EXPECT_TRUE((test_uniform_ctor<float16, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<float16, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<bfloat16, float>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, double>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, float16>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int8_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int8_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int16_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int16_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int32_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int32_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<int64_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<int64_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint8_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint8_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint16_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint16_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint32_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint32_t, uint64_t>()));

    EXPECT_TRUE((test_uniform_ctor<uint64_t, float>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, double>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, float16>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, bfloat16>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, int64_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint8_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint16_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint32_t>()));
    EXPECT_TRUE((test_uniform_ctor<uint64_t, uint64_t>()));
}

TEST(constant, bad_get_data_ptr)
{
    op::Constant c(element::f32, Shape{}, vector<float>{1.0});
    EXPECT_EQ(*c.get_data_ptr<element::Type_t::f32>(), 1.0);
    try
    {
        c.get_data_ptr<element::Type_t::f64>();
        FAIL() << "Bad type not detected.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
    try
    {
        c.get_data_ptr<element::Type_t::i32>();
        FAIL() << "Bad type not detected.";
    }
    catch (const CheckFailure& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
}
