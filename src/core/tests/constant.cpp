// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"

namespace ov {
namespace test {

struct TestDType {
    TestDType(float v) : value{v} {}

    template <typename I>
    explicit TestDType(I v) : value{static_cast<float>(v)} {}

    operator float() const {
        return value;
    }

    // To print values in tests
    friend std::ostream& operator<<(std::ostream& os, const TestDType& obj) {
        os << obj.value;
        return os;
    }

    float value;
};

using std::string;
using std::vector;

using testing::Each;
using testing::ElementsAre;
using testing::HasSubstr;

//
// boolean
//

TEST(constant, boolean_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::boolean, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }

    EXPECT_EQ(c.get_strides(), Strides({1}));
}

TEST(constant, boolean_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::boolean, shape, vector<string>{"1"});
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

TEST(constant, boolean_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::boolean, shape, vector<char>{1, 0, 1, 0});
    auto v = c.get_vector<char>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_vector<char>();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 1);
    EXPECT_EQ(p[3], 0);
}

TEST(constant, boolean_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::boolean, shape, vector<char>{1});
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

TEST(constant, float_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::f32, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }

    EXPECT_EQ(c.get_strides(), Strides({4}));
}

TEST(constant, float_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f32, shape, vector<string>{"1"});
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

TEST(constant, float_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f32, shape, vector<float>{1, 0, 1, 0});
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

TEST(constant, float_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f32, shape, vector<float>{1});
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

TEST(constant, double_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::f64, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::f64.size()}));
}

TEST(constant, double_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f64, shape, vector<string>{"1"});
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

TEST(constant, double_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f64, shape, vector<double>{1, 0, 1, 0});
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

TEST(constant, double_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f64, shape, vector<double>{1});
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
// int4
//

TEST(constant, int4_string) {
    Shape shape{3};
    std::vector<std::string> input{"1", "0", "-1"};
    ov::op::v0::Constant c(element::i4, shape, input);
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], -1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x01, p[0]);
    EXPECT_EQ(0x0F, p[1]);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_THROW(c.get_strides(), Exception);
}

TEST(constant, int4_string_broadcast_negative_number) {
    Shape shape{3};
    ov::op::v0::Constant c(element::i4, shape, vector<string>{"-1"});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], -1);
    EXPECT_EQ(v[1], -1);
    EXPECT_EQ(v[2], -1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xFF, p[0]);
    EXPECT_EQ(0x0F, p[1]);

    EXPECT_EQ(std::vector<std::string>(3, "-1"), c.get_value_strings());
}

TEST(constant, int4_string_broadcast_positive_number) {
    Shape shape{3};
    ov::op::v0::Constant c(element::i4, shape, vector<string>{"1"});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x11, p[0]);
    EXPECT_EQ(0x01, p[1]);

    EXPECT_EQ(std::vector<std::string>(3, "1"), c.get_value_strings());
}

TEST(constant, int4_vector_negative_number) {
    Shape shape{3};
    ov::op::v0::Constant c(element::i4, shape, vector<int8_t>{-1, -2, -1});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(-1));
    EXPECT_EQ(v[1], int8_t(-2));
    EXPECT_EQ(v[2], int8_t(-1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xEF, p[0]);
    EXPECT_EQ(0x0F, p[1]);
}

TEST(constant, int4_vector_positive_number) {
    Shape shape{3};
    ov::op::v0::Constant c(element::i4, shape, vector<int8_t>{1, 2, 5});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(1));
    EXPECT_EQ(v[1], int8_t(2));
    EXPECT_EQ(v[2], int8_t(5));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x21, p[0]);
    EXPECT_EQ(0x05, p[1]);
}

TEST(constant, int4_vector_broadcast_negative_number) {
    Shape shape{3};
    ov::op::v0::Constant c(element::i4, shape, vector<int8_t>{-1});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(-1));
    EXPECT_EQ(v[1], int8_t(-1));
    EXPECT_EQ(v[2], int8_t(-1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xFF, p[0]);
    EXPECT_EQ(0x0F, p[1]);
}

TEST(constant, int4_vector_broadcast_positive_number) {
    Shape shape{3};
    ov::op::v0::Constant c(element::i4, shape, vector<int8_t>{3});
    auto v = c.cast_vector<int8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(3));
    EXPECT_EQ(v[1], int8_t(3));
    EXPECT_EQ(v[2], int8_t(3));

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(0x33, p[0]);
    EXPECT_EQ(0x03, p[1]);
}

TEST(constant, int4_input_value_validation) {
    Shape shape{2};
    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, 8), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, -9), ::ov::AssertFailure);

    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, std::vector<int>{-9}), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, std::vector<int>{8}), ::ov::AssertFailure);

    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, std::vector<int>{-9, 1}), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, std::vector<int>{8, 2}), ::ov::AssertFailure);

    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, std::vector<std::string>{"-9", "1"}), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::i4, shape, std::vector<std::string>{"8", "1"}), ::ov::AssertFailure);
}

TEST(constant, int4_write_then_cast_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{{1.0f}, {-2.0f}, {7.0f}};
    ov::op::v0::Constant c(element::i4, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, input);
}
//
// int8
//

TEST(constant, int8_string) {
    Shape shape{4};
    std::vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::i8, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::i8.size()}));
}

TEST(constant, int8_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i8, shape, vector<string>{"1"});
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

    EXPECT_EQ(std::vector<std::string>(4, "1"), c.get_value_strings());
}

TEST(constant, int8_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i8, shape, vector<int8_t>{1, 0, 1, 0});
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

TEST(constant, int8_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i8, shape, vector<int8_t>{1});
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

TEST(constant, int16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::i16, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::i16.size()}));
}

TEST(constant, int16_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i16, shape, vector<string>{"1"});
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

TEST(constant, int16_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i16, shape, vector<int16_t>{1, 0, 1, 0});
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

TEST(constant, int16_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i16, shape, vector<int16_t>{1});
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

TEST(constant, int32_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::i32, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int32_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i32, shape, vector<string>{"1"});
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

TEST(constant, int32_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i32, shape, vector<int32_t>{1, 0, 1, 0});
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

TEST(constant, int32_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i32, shape, vector<int32_t>{1});
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

TEST(constant, int64_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::i64, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, int64_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i64, shape, vector<string>{"1"});
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

TEST(constant, int64_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i64, shape, vector<int64_t>{1, 0, 1, 0});
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

TEST(constant, int64_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::i64, shape, vector<int64_t>{1});
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

TEST(constant, int64_string_max) {
    Shape shape{4};
    vector<string> input{"9223372036854775807", "9223372036854775807", "9223372036854775807", "9223372036854775807"};

    constexpr auto exp_value = std::numeric_limits<int64_t>::max();
    ov::op::v0::Constant c(element::i64, shape, input);
    auto v = c.get_vector<int64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, testing::Each(exp_value));

    const auto p = c.get_data_ptr<int64_t>();
    EXPECT_EQ(p[0], exp_value);
    EXPECT_EQ(p[1], exp_value);
    EXPECT_EQ(p[2], exp_value);
    EXPECT_EQ(p[3], exp_value);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

//
// uint1
//

TEST(constant, uint1_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::u1, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0xF0, 0b10100000);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_THROW(c.get_strides(), Exception);
}

TEST(constant, uint1_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u1, shape, vector<string>{"1"});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11110000, 0b11110000);
}

TEST(constant, uint1_vector_less_than_single_byte) {
    Shape shape{4};
    vector<uint8_t> input{1, 0, 1, 0};
    ov::op::v0::Constant c(element::u1, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(v[i], input[i]) << "Error on index: " << i;
    }

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11110000, 0b10100000);
}

TEST(constant, uint1_vector_bigger_than_single_byte) {
    Shape shape{12};
    vector<uint8_t> input{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    ov::op::v0::Constant c(element::u1, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(v[i], input[i]) << "Error on index: " << i;
    }

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11110000, 0b10100000);
}

TEST(constant, uint1_vector_broadcast) {
    Shape shape{3};
    ov::op::v0::Constant c(element::u1, shape, vector<int8_t>{1});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(1));
    EXPECT_EQ(v[1], int8_t(1));
    EXPECT_EQ(v[2], int8_t(1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0xE0, p[0] & 0xE0);
}

TEST(constant, uint1_write_then_cast_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{{1.0f}, {0.0f}, {12.0f}};
    ov::op::v0::Constant c(element::u1, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, std::vector<TestDType>({1.0f, 0.0f, 1.0f}));
}

//
// uint2
//

TEST(constant, uint2_string) {
    const auto shape = Shape{4};

    op::v0::Constant c(element::u2, shape, vector<string>{"3", "0", "1", "2"});
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(3, 0, 1, 2));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b11000110);

    EXPECT_EQ(c.convert_value_to_string(1), "0");
    EXPECT_THAT(c.get_value_strings(), ElementsAre("3", "0", "1", "2"));
    EXPECT_THROW(c.get_strides(), Exception);
}

TEST(constant, uint2_string_broadcast) {
    const auto shape = Shape{5};

    op::v0::Constant c(element::u2, shape, vector<string>{"1"});
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b01010101);
    EXPECT_EQ(p[1] & 0b11000000, 0b01000000);
}

TEST(constant, uint2_vector_less_than_single_byte) {
    auto const shape = Shape{3};
    const auto input = std::vector<uint8_t>{2, 3, 1};

    op::v0::Constant c(element::u2, shape, input);
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(2, 3, 1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11111100, 0b10110100);
}

TEST(constant, uint2_vector_bigger_than_single_byte) {
    auto const shape = Shape{7};
    const auto input = std::vector<uint8_t>{2, 3, 1, 0, 1, 2, 0};

    op::v0::Constant c(element::u2, shape, input);
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(2, 3, 1, 0, 1, 2, 0));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b10110100);
    EXPECT_EQ(p[1] & 0b11111100, 0b01100000);
}

TEST(constant, uint2_vector_broadcast) {
    const auto shape = Shape{3};
    op::v0::Constant c(element::u2, shape, vector<int8_t>{2});

    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(2));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0] & 0b11111100, 0b10101000);
}

TEST(constant, uint2_write_then_cast_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{{1.0f}, {3.0f}, {2.0f}};
    ov::op::v0::Constant c(element::u2, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, input);
}

TEST(constant, uint2_input_value_validation) {
    const auto shape = Shape{2};
    const auto exp_sub_str = "out of range for u2";

    OV_EXPECT_THROW(op::v0::Constant c(element::u2, shape, -1), AssertFailure, HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u2, shape, 4), AssertFailure, HasSubstr(exp_sub_str));

    OV_EXPECT_THROW(op::v0::Constant c(element::u2, shape, std::vector<int>{1, -2}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u2, shape, std::vector<int>{0, 4}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));

    OV_EXPECT_THROW(op::v0::Constant c(element::u2, shape, std::vector<std::string>{"-1", "3"}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u2, shape, std::vector<std::string>{"4", "1"}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
}

//
// uint3
//

TEST(constant, uint3_string) {
    const auto shape = Shape{8};

    op::v0::Constant c(element::u3, shape, vector<string>{"3", "0", "1", "2", "4", "7", "5", "6"});
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(3, 0, 1, 2, 4, 7, 5, 6));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b11000110);
    EXPECT_EQ(p[1], 0b00110110);
    EXPECT_EQ(p[2], 0b00001111);

    EXPECT_EQ(c.convert_value_to_string(6), "5");
    EXPECT_THAT(c.get_value_strings(), ElementsAre("3", "0", "1", "2", "4", "7", "5", "6"));
    EXPECT_THROW(c.get_strides(), Exception);
}

TEST(constant, uint3_string_broadcast) {
    const auto shape = Shape{5};

    op::v0::Constant c(element::u3, shape, vector<string>{"5"});
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(5));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b01010101);
    EXPECT_EQ(p[1], 0b01000000);
    EXPECT_EQ(p[2], 0b11111000);
}

TEST(constant, uint3_vector_less_than_one_storage_unit) {
    auto const shape = Shape{3};
    const auto input = std::vector<uint8_t>{5, 3, 1};

    op::v0::Constant c(element::u3, shape, input);
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(5, 3, 1));

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(p[0], 0b01110100);
    EXPECT_EQ(p[1], 0);
    EXPECT_EQ(p[2], 0b10000000);
}

TEST(constant, uint3_vector_greater_than_one_storage_unit) {
    auto const shape = Shape{10};
    const auto input = std::vector<uint8_t>{2, 3, 1, 0, 4, 5, 6, 7, 5, 2};

    op::v0::Constant c(element::u3, shape, input);
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(2, 3, 1, 0, 4, 5, 6, 7, 5, 2));

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(p[0], 0b10110100);
    EXPECT_EQ(p[1], 0b00011011);
    EXPECT_EQ(p[2], 0b00001111);

    EXPECT_EQ(p[3], 0b01100000);
    EXPECT_EQ(p[4], 0);
    EXPECT_EQ(p[5], 0b10000000);
}

TEST(constant, uint3_vector_broadcast) {
    const auto shape = Shape{8};
    op::v0::Constant c(element::u3, shape, vector<int8_t>{2});

    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(2));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0b10101010);
    EXPECT_EQ(p[1], 0b10101010);
    EXPECT_EQ(p[2], 0b00000000);
}

TEST(constant, uint3_write_then_cast_custom_type) {
    Shape shape{5};
    std::vector<TestDType> input{{1.0f}, {3.0f}, {2.0f}, {6.1f}, {3.5f}};
    ov::op::v0::Constant c(element::u3, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, std::vector<TestDType>({1.0f, 3.0f, 2.0f, 6.0f, 3.0f}));
}

TEST(constant, uint3_input_value_validation) {
    const auto shape = Shape{2};
    const auto exp_sub_str = "out of range for u3";

    OV_EXPECT_THROW(op::v0::Constant c(element::u3, shape, -1), AssertFailure, HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u3, shape, 8), AssertFailure, HasSubstr(exp_sub_str));

    OV_EXPECT_THROW(op::v0::Constant c(element::u3, shape, std::vector<int>{1, -2}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u3, shape, std::vector<int>{0, 8}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));

    OV_EXPECT_THROW(op::v0::Constant c(element::u3, shape, std::vector<std::string>{"-1", "3"}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u3, shape, std::vector<std::string>{"9", "1"}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
}

//
// uint4
//

TEST(constant, uint4_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::u4, shape, input);
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x01);
    EXPECT_EQ(p[1], 0x01);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_THROW(c.get_strides(), Exception);
}

TEST(constant, uint4_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u4, shape, vector<string>{"1"});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 1);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 1);

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x11);
    EXPECT_EQ(p[1], 0x11);
}

TEST(constant, uint4_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u4, shape, vector<uint8_t>{1, 0, 1, 0});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], 1);
    EXPECT_EQ(v[1], 0);
    EXPECT_EQ(v[2], 1);
    EXPECT_EQ(v[3], 0);

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(p[0], 0x01);
    EXPECT_EQ(p[1], 0x01);
}

TEST(constant, uint4_vector_broadcast) {
    Shape shape{3};
    ov::op::v0::Constant c(element::u4, shape, vector<uint8_t>{1});
    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], int8_t(1));
    EXPECT_EQ(v[1], int8_t(1));
    EXPECT_EQ(v[2], int8_t(1));

    const auto p = c.get_data_ptr<uint8_t>();
    const auto first_byte = p[0];
    const auto second_byte = p[1];
    EXPECT_EQ(0x11, first_byte);
    EXPECT_EQ(0x01, second_byte);

    const auto vector = c.get_vector<uint8_t>();
    EXPECT_EQ(vector[0], 0x11);
    EXPECT_EQ(vector[1], 0x01);
}

TEST(constant, uint4_get_vector_from_one_element) {
    auto c = std::make_shared<op::v0::Constant>(element::u4, Shape{1}, 9);
    auto v = c->get_vector<uint8_t>();

    ASSERT_EQ(v.size(), 1);
    EXPECT_EQ(v[0], 0x09);
}

TEST(constant, uint4_get_vector_from_scalar) {
    auto c = std::make_shared<op::v0::Constant>(element::u4, Shape{}, 8);
    auto v = c->get_vector<uint8_t>();

    ASSERT_EQ(v.size(), 1);
    EXPECT_EQ(v[0], 0x08);
}

TEST(constant, uint4_input_value_validation) {
    Shape shape{2};
    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, 16), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, -1), ::ov::AssertFailure);

    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, std::vector<int>{-1}), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, std::vector<int>{16}), ::ov::AssertFailure);

    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, std::vector<int>{-1, 1}), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, std::vector<int>{16, 2}), ::ov::AssertFailure);

    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, std::vector<std::string>{"-1", "1"}), ::ov::AssertFailure);
    EXPECT_THROW(ov::op::v0::Constant c(element::u4, shape, std::vector<std::string>{"16", "1"}), ::ov::AssertFailure);
}

TEST(constant, uint4_write_then_cast_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{{1.0f}, {3.0f}, {12.0f}};
    ov::op::v0::Constant c(element::u4, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, input);
}

//
// uint6
//

TEST(constant, uint6_string) {
    const auto shape = Shape{4};

    op::v0::Constant c(element::u6, shape, vector<string>{"4", "9", "15", "16"});
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(4, 9, 15, 16));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x49);
    EXPECT_EQ(p[1], 0xf0);
    EXPECT_EQ(p[2], 0b00000001);

    EXPECT_EQ(c.convert_value_to_string(2), "15");
    EXPECT_THAT(c.get_value_strings(), ElementsAre("4", "9", "15", "16"));
    EXPECT_THROW(c.get_strides(), Exception);
}

TEST(constant, uint6_string_broadcast) {
    const auto shape = Shape{4};

    op::v0::Constant c(element::u6, shape, vector<string>{"5"});
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(5));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x55);
    EXPECT_EQ(p[1], 0x55);
    EXPECT_EQ(p[2], 0b00000000);
}

TEST(constant, uint6_vector_less_than_one_storage_unit) {
    auto const shape = Shape{3};
    const auto input = std::vector<uint8_t>{5, 23, 1};

    op::v0::Constant c(element::u6, shape, input);
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(5, 23, 1));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x57);
    EXPECT_EQ(p[1], 0x10);
    EXPECT_EQ(p[2], 0b00010000);
}

TEST(constant, uint6_vector_greater_than_one_storage_unit) {
    auto const shape = Shape{6};
    const auto input = std::vector<uint8_t>{25, 3, 1, 0, 45, 5};

    op::v0::Constant c(element::u6, shape, input);
    auto v = c.cast_vector<uint8_t>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, ElementsAre(25, 3, 1, 0, 45, 5));

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(p[0], 0x93);
    EXPECT_EQ(p[1], 0x10);
    EXPECT_EQ(p[2], 0b01000000);

    EXPECT_EQ(p[3], 0xd5);
    EXPECT_EQ(p[4], 0);
    EXPECT_EQ(p[5], 0b10000000);
}

TEST(constant, uint6_vector_broadcast) {
    const auto shape = Shape{4};
    op::v0::Constant c(element::u6, shape, vector<int8_t>{45});

    auto v = c.cast_vector<uint8_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(45));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0xdd);
    EXPECT_EQ(p[1], 0xdd);
    EXPECT_EQ(p[2], 0b10101010);
}

TEST(constant, uint6_write_then_cast_custom_type) {
    Shape shape{5};
    std::vector<TestDType> input{{1.0f}, {3.0f}, {2.0f}, {6.1f}, {3.5f}};
    ov::op::v0::Constant c(element::u6, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, std::vector<TestDType>({1.0f, 3.0f, 2.0f, 6.0f, 3.0f}));
}

TEST(constant, uint6_input_value_validation) {
    const auto shape = Shape{2};
    const auto exp_sub_str = "out of range for u6";

    OV_EXPECT_THROW(op::v0::Constant c(element::u6, shape, -1), AssertFailure, HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u6, shape, 64), AssertFailure, HasSubstr(exp_sub_str));

    OV_EXPECT_THROW(op::v0::Constant c(element::u6, shape, std::vector<int>{1, -2}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u6, shape, std::vector<int>{0, 64}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));

    OV_EXPECT_THROW(op::v0::Constant c(element::u6, shape, std::vector<std::string>{"-1", "3"}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
    OV_EXPECT_THROW(op::v0::Constant c(element::u6, shape, std::vector<std::string>{"65", "1"}),
                    AssertFailure,
                    HasSubstr(exp_sub_str));
}

//
// uint8
//

TEST(constant, uint8_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::u8, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::u8.size()}));
}

TEST(constant, uint8_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u8, shape, vector<string>{"1"});
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

TEST(constant, uint8_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u8, shape, vector<uint8_t>{1, 0, 1, 0});
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

TEST(constant, uint8_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u8, shape, vector<uint8_t>{1});
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

TEST(constant, uint16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::u16, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::u16.size()}));
}

TEST(constant, uint16_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u16, shape, vector<string>{"1"});
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

TEST(constant, uint16_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u16, shape, vector<uint16_t>{1, 0, 1, 0});
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

TEST(constant, uint16_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u16, shape, vector<uint16_t>{1});
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

TEST(constant, uint32_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::u32, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::u32.size()}));
}

TEST(constant, uint32_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u32, shape, vector<string>{"1"});
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

TEST(constant, uint32_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u32, shape, vector<uint32_t>{1, 0, 1, 0});
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

TEST(constant, uint32_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u32, shape, vector<uint32_t>{1});
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

TEST(constant, uint64_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::u64, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::u64.size()}));
}

TEST(constant, uint64_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u64, shape, vector<string>{"1"});
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

TEST(constant, uint64_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u64, shape, vector<uint64_t>{1, 0, 1, 0});
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

TEST(constant, uint64_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::u64, shape, vector<uint64_t>{1});
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

TEST(constant, uint64_string_max) {
    Shape shape{4};
    vector<string> input{"18446744073709551615",
                         "18446744073709551615",
                         "18446744073709551615",
                         "18446744073709551615"};
    ov::op::v0::Constant c(element::u64, shape, input);
    constexpr auto exp_value = std::numeric_limits<uint64_t>::max();
    auto v = c.get_vector<uint64_t>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, testing::Each(exp_value));

    const auto p = c.get_data_ptr<uint64_t>();
    EXPECT_EQ(p[0], exp_value);
    EXPECT_EQ(p[1], exp_value);
    EXPECT_EQ(p[2], exp_value);
    EXPECT_EQ(p[3], exp_value);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

//
// nf4
//
TEST(constant, nf4_write_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{{-1.1f}, {-.5f}, {2.0f}};
    ov::op::v0::Constant c(element::nf4, shape, input);

    auto p = c.get_data_ptr<uint8_t>();

    EXPECT_EQ(p[0], 0x20);
    EXPECT_EQ(p[1], 0x0f);
    EXPECT_THROW(c.get_strides(), Exception);
}

//
// bfloat16
//

TEST(constant, bfloat16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::bf16, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::bf16.size()}));
}

TEST(constant, bfloat16_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::bf16, shape, vector<string>{"1"});
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

TEST(constant, bfloat16_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::bf16, shape, vector<bfloat16>{1, 0, 1, 0});
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

TEST(constant, bfloat16_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::bf16, shape, vector<bfloat16>{1});
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

TEST(constant, float16_string) {
    Shape shape{4};
    vector<string> input{"1", "0", "1", "0"};
    ov::op::v0::Constant c(element::f16, shape, input);
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

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::f16.size()}));
}

TEST(constant, float16_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f16, shape, vector<string>{"1"});
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

TEST(constant, float16_vector) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f16, shape, vector<float16>{1, 0, 1, 0});
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

TEST(constant, float8_e4m3_vector) {
    const auto data_vec = std::vector<ov::float8_e4m3>{std::numeric_limits<ov::float8_e4m3>::lowest(),
                                                       -std::numeric_limits<ov::float8_e4m3>::min(),
                                                       std::numeric_limits<ov::float8_e4m3>::min(),
                                                       std::numeric_limits<ov::float8_e4m3>::max(),
                                                       std::numeric_limits<ov::float8_e4m3>::denorm_min(),
                                                       -1.5f,
                                                       -1.f,
                                                       -0.5f,
                                                       0.f,
                                                       0.5f,
                                                       1.f,
                                                       1.5f};
    Shape data_shape{data_vec.size()};
    EXPECT_EQ(data_vec.size(), shape_size(data_shape));

    ov::op::v0::Constant const_op_from_vec(ov::element::f8e4m3, data_shape, data_vec);
    EXPECT_EQ(data_vec, const_op_from_vec.get_vector<ov::float8_e4m3>());

    ov::op::v0::Constant const_op_from_ptr(ov::element::f8e4m3, data_shape, data_vec.data());
    EXPECT_EQ(data_vec, const_op_from_ptr.get_vector<ov::float8_e4m3>());
    EXPECT_EQ(const_op_from_ptr.get_strides(), Strides({element::f8e4m3.size()}));
}

TEST(constant, float8_e5m3_vector) {
    const auto data_vec = std::vector<ov::float8_e5m2>{std::numeric_limits<ov::float8_e5m2>::lowest(),
                                                       -std::numeric_limits<ov::float8_e5m2>::min(),
                                                       std::numeric_limits<ov::float8_e5m2>::min(),
                                                       std::numeric_limits<ov::float8_e5m2>::max(),
                                                       std::numeric_limits<ov::float8_e5m2>::denorm_min(),
                                                       -1.5f,
                                                       -1.f,
                                                       -0.5f,
                                                       0.f,
                                                       0.5f,
                                                       1.f,
                                                       1.5f};
    Shape data_shape{data_vec.size()};
    EXPECT_EQ(data_vec.size(), shape_size(data_shape));

    ov::op::v0::Constant const_op_from_vec(ov::element::f8e5m2, data_shape, data_vec);
    EXPECT_EQ(data_vec, const_op_from_vec.get_vector<ov::float8_e5m2>());

    ov::op::v0::Constant const_op_from_ptr(ov::element::f8e5m2, data_shape, data_vec.data());
    EXPECT_EQ(data_vec, const_op_from_ptr.get_vector<ov::float8_e5m2>());
    EXPECT_EQ(const_op_from_ptr.get_strides(), Strides({element::f8e5m2.size()}));
}

TEST(constant, float8_e8m0_vector) {
    const auto data_vec = std::vector<ov::float8_e8m0>{std::numeric_limits<ov::float8_e8m0>::lowest(),
                                                       -std::numeric_limits<ov::float8_e8m0>::min(),
                                                       std::numeric_limits<ov::float8_e8m0>::min(),
                                                       std::numeric_limits<ov::float8_e8m0>::max(),
                                                       std::numeric_limits<ov::float8_e8m0>::min(),
                                                       -1.5f,
                                                       -1.f,
                                                       -0.5f,
                                                       0.f,
                                                       0.5f,
                                                       1.f,
                                                       1.5f};
    Shape data_shape{data_vec.size()};
    EXPECT_EQ(data_vec.size(), shape_size(data_shape));

    ov::op::v0::Constant const_op_from_vec(ov::element::f8e8m0, data_shape, data_vec);
    EXPECT_EQ(data_vec, const_op_from_vec.get_vector<ov::float8_e8m0>());

    ov::op::v0::Constant const_op_from_ptr(ov::element::f8e8m0, data_shape, data_vec.data());
    EXPECT_EQ(data_vec, const_op_from_ptr.get_vector<ov::float8_e8m0>());
}

TEST(constant, float16_vector_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::f16, shape, vector<float16>{1});
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

TEST(constant, shared_data) {
    Shape shape{100, 200};
    auto c1 = std::make_shared<ov::op::v0::Constant>(element::f16, shape, vector<float16>{123});
    auto c2 = std::static_pointer_cast<ov::op::v0::Constant>(c1->clone_with_new_inputs({}));
    const int16_t* p1 = c1->get_data_ptr<int16_t>();
    const int16_t* p2 = c2->get_data_ptr<int16_t>();
    EXPECT_EQ(p1, p2);
}

//
// string
//

TEST(constant, ov_string) {
    Shape shape{4};
    vector<std::string> input{"abc", "one two three", "1", "0"};
    ov::op::v0::Constant c(element::string, shape, input);
    auto v = c.get_vector<std::string>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], "abc");
    EXPECT_EQ(v[1], "one two three");
    EXPECT_EQ(v[2], "1");
    EXPECT_EQ(v[3], "0");

    const std::string* p = c.get_data_ptr<std::string>();
    EXPECT_EQ(p[0], "abc");
    EXPECT_EQ(p[1], "one two three");
    EXPECT_EQ(p[2], "1");
    EXPECT_EQ(p[3], "0");

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
    EXPECT_EQ(c.get_strides(), Strides({element::string.size()}));
}

TEST(constant, ov_string_broadcast) {
    Shape shape{4};
    ov::op::v0::Constant c(element::string, shape, vector<string>{"one two "});
    auto v = c.get_vector<std::string>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v[0], "one two ");
    EXPECT_EQ(v[1], "one two ");
    EXPECT_EQ(v[2], "one two ");
    EXPECT_EQ(v[3], "one two ");

    const std::string* p = c.get_data_ptr<std::string>();
    EXPECT_EQ(p[0], "one two ");
    EXPECT_EQ(p[1], "one two ");
    EXPECT_EQ(p[2], "one two ");
    EXPECT_EQ(p[3], "one two ");
}

TEST(constant, ov_string_shared_data) {
    Shape shape{100, 200};
    auto c1 = std::make_shared<ov::op::v0::Constant>(element::string, shape, vector<std::string>{"123"});
    auto c2 = std::static_pointer_cast<ov::op::v0::Constant>(c1->clone_with_new_inputs({}));
    const int16_t* p1 = c1->get_data_ptr<int16_t>();
    const int16_t* p2 = c2->get_data_ptr<int16_t>();
    EXPECT_EQ(p1, p2);
}

TEST(constant, ov_string_broadcast_from_non_string) {
    EXPECT_THROW(std::ignore = op::v0::Constant::create(element::string, Shape{4}, std::vector<int>{10}), Exception);
}

TEST(constant, ov_string_from_non_string_vector) {
    EXPECT_THROW(std::ignore = op::v0::Constant::create(element::string, Shape{4}, std::vector<int>{10, 1, 3, 2}),
                 Exception);
}

//
// f4e2m1
//
TEST(constant, f4e2m1_string) {
    vector<string> input{"1", "0", "1.5", "2"};
    ov::op::v0::Constant c(element::f4e2m1, Shape{4}, input);
    auto v = c.cast_vector<float4_e2m1>();
    ASSERT_EQ(v.size(), shape_size(c.get_shape()));
    EXPECT_THAT(v, ElementsAre(1.0f, 0.0f, 1.5f, 2.0f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x02);
    EXPECT_EQ(p[1], 0x43);

    EXPECT_EQ(input, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(input[i], c.convert_value_to_string(i));
    }
}

TEST(constant, f4e2m1_string_broadcast) {
    Shape shape{4};
    op::v0::Constant c(element::f4e2m1, shape, std::vector<string>{"1.5"});
    auto v = c.cast_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(1.5f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x33);
    EXPECT_EQ(p[1], 0x33);
}

TEST(constant, f4e2m1_vector) {
    op::v0::Constant c(element::f4e2m1, Shape{5}, std::vector<float4_e2m1>{-1.5f, 4.0f, -2.0f, 1.5f, -3.0f});
    auto v = c.cast_vector<float>();
    EXPECT_THAT(v, ElementsAre(-1.5f, 4.0f, -2.0f, 1.5f, -3.0f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x6b);
    EXPECT_EQ(p[1], 0x3c);
    EXPECT_EQ(p[2], 0x0d);
}

TEST(constant, f4e2m1_from_float_vector) {
    op::v0::Constant c(element::f4e2m1, Shape{5}, std::vector<float>{-1.5f, 4.0f, -2.0f, 1.5f, -3.0f});
    auto v = c.cast_vector<float>();
    EXPECT_THAT(v, ElementsAre(-1.5f, 4.0f, -2.0f, 1.5f, -3.0f));

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(p[0], 0x6b);
    EXPECT_EQ(p[1], 0x3c);
    EXPECT_EQ(p[2], 0x0d);
}

TEST(constant, f4e2m1_vector_broadcast) {
    Shape shape{3};
    op::v0::Constant c(element::f4e2m1, shape, std::vector<float4_e2m1>{1.5f});
    auto v = c.cast_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(1.5f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x33, p[0]);
    EXPECT_EQ(0x03, p[1]);
}

TEST(constant, f4e2m1_write_then_cast_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{1.5f, 3.0f, 6.0f};
    op::v0::Constant c(element::f4e2m1, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, input);
}

//
// f8e8m0
//
TEST(constant, f8e8m0_string) {
    vector<string> input{"1", "0", "4", "0.5"};
    vector<string> output{"1", "5.87747e-39", "4", "0.5"};
    ov::op::v0::Constant c(element::f8e8m0, Shape{4}, input);
    auto v = c.cast_vector<float8_e8m0>();
    ASSERT_EQ(v.size(), shape_size(c.get_shape()));
    EXPECT_THAT(v, ElementsAre(1.0f, std::numeric_limits<float>::min() / 2, 4.0f, 0.5f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x7f);
    EXPECT_EQ(p[1], 0x00);
    EXPECT_EQ(p[2], 0x81);
    EXPECT_EQ(p[3], 0x7e);

    EXPECT_EQ(output, c.get_value_strings());

    for (unsigned i = 0; i != input.size(); ++i) {
        EXPECT_EQ(output[i], c.convert_value_to_string(i));
    }
}

TEST(constant, f8e8m0_string_broadcast) {
    Shape shape{4};
    op::v0::Constant c(element::f8e8m0, shape, std::vector<string>{"1.5"});
    auto v = c.cast_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(2.0f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x80);
    EXPECT_EQ(p[1], 0x80);
    EXPECT_EQ(p[2], 0x80);
    EXPECT_EQ(p[3], 0x80);
}

TEST(constant, f8e8m0_vector) {
    op::v0::Constant c(element::f8e8m0, Shape{5}, std::vector<float8_e8m0>{-1.5f, 4.0f, 2.0f, 1.5f, 3.0f});
    auto v = c.cast_vector<float>();
    EXPECT_THAT(v, ElementsAre(std::numeric_limits<float>::min() / 2, 4.0f, 2.0f, 2.0f, 2.0f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(p[0], 0x00);
    EXPECT_EQ(p[1], 0x81);
    EXPECT_EQ(p[2], 0x80);
    EXPECT_EQ(p[3], 0x80);
    EXPECT_EQ(p[4], 0x80);
}

TEST(constant, f8e8m0_from_float_vector) {
    op::v0::Constant c(element::f8e8m0, Shape{5}, std::vector<float>{-1.5f, 4.0f, 2.0f, 1.5f, 2.0f});
    auto v = c.cast_vector<float>();
    EXPECT_THAT(v, ElementsAre(std::numeric_limits<float>::min() / 2, 4.0f, 2.0f, 2.0f, 2.0f));

    const auto p = c.get_vector<uint8_t>();
    EXPECT_EQ(p[0], 0x00);
    EXPECT_EQ(p[1], 0x81);
    EXPECT_EQ(p[2], 0x80);
    EXPECT_EQ(p[3], 0x80);
    EXPECT_EQ(p[4], 0x80);
}

TEST(constant, f8e8m0_vector_broadcast) {
    Shape shape{3};
    op::v0::Constant c(element::f8e8m0, shape, std::vector<float8_e8m0>{1.5f});
    auto v = c.cast_vector<float>();
    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_THAT(v, Each(2.0f));

    const auto p = c.get_data_ptr<uint8_t>();
    EXPECT_EQ(0x80, p[0]);
    EXPECT_EQ(0x80, p[1]);
}

TEST(constant, f8e8m0_write_then_cast_custom_type) {
    Shape shape{3};
    std::vector<TestDType> input{1.5f, 3.0f, 6.0f};
    std::vector<TestDType> expected{2.0f, 2.0f, 8.0f};
    op::v0::Constant c(element::f8e8m0, shape, input);

    auto v = c.cast_vector<TestDType>();

    ASSERT_EQ(v.size(), shape_size(shape));
    EXPECT_EQ(v, expected);
}

template <typename T1, typename T2>
::testing::AssertionResult test_convert(vector<T1> expected = {1, 2, 3, 4, 6}) {
    Shape shape{5};
    auto c1 = std::make_shared<ov::op::v0::Constant>(ov::element::from<T2>(), shape, expected);
    vector<T1> actual = c1->template cast_vector<T1>();
    ::testing::AssertionResult rc =
        (actual == expected ? ::testing::AssertionSuccess() : ::testing::AssertionFailure());
    rc << "Conversion failed";
    return rc;
}

TEST(constant, convert_input_ov_string) {
    Shape shape{5};
    vector<std::string> expected{"1", "2", "3", "4", "5"};
    auto c1 = std::make_shared<ov::op::v0::Constant>(ov::element::from<std::string>(), shape, expected);
    vector<std::string> actual = c1->template cast_vector<std::string>();

    EXPECT_EQ(actual, expected);
}

TEST(constant, convert_input) {
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
    EXPECT_TRUE((test_convert<float, float4_e2m1>()));
    EXPECT_TRUE((test_convert<float, float8_e8m0>({1, 2, 4, 8, 16})));

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
    EXPECT_TRUE((test_convert<double, float4_e2m1>()));
    EXPECT_TRUE((test_convert<double, float8_e8m0>({1, 2, 4, 8, 16})));

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
    EXPECT_TRUE((test_convert<float16, float4_e2m1>()));
    EXPECT_TRUE((test_convert<float16, float8_e8m0>({1, 2, 4, 8, 16})));

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
    EXPECT_TRUE((test_convert<bfloat16, float4_e2m1>()));
    EXPECT_TRUE((test_convert<bfloat16, float8_e8m0>({1, 2, 4, 8, 16})));

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
::testing::AssertionResult test_uniform_ctor() {
    Shape shape{5};
    vector<T1> expected{2, 2, 2, 2, 2};
    auto c1 = std::make_shared<ov::op::v0::Constant>(ov::element::from<T2>(), shape, 2);
    vector<T1> actual = c1->template cast_vector<T1>();
    ::testing::AssertionResult rc =
        (actual == expected ? ::testing::AssertionSuccess() : ::testing::AssertionFailure());
    rc << "Construction of uniform Constant failed";
    return rc;
}

TEST(constant, construct_uniform) {
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
    EXPECT_TRUE((test_uniform_ctor<float, float4_e2m1>()));
    EXPECT_TRUE((test_uniform_ctor<float, float8_e8m0>()));

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
    EXPECT_TRUE((test_uniform_ctor<double, float4_e2m1>()));
    EXPECT_TRUE((test_uniform_ctor<double, float8_e8m0>()));

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
    EXPECT_TRUE((test_uniform_ctor<float16, float4_e2m1>()));
    EXPECT_TRUE((test_uniform_ctor<float16, float8_e8m0>()));

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
    EXPECT_TRUE((test_uniform_ctor<bfloat16, float4_e2m1>()));
    EXPECT_TRUE((test_uniform_ctor<bfloat16, float8_e8m0>()));

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

TEST(constant, bad_get_data_ptr) {
    ov::op::v0::Constant c(element::f32, Shape{}, vector<float>{1.0});
    EXPECT_EQ(*c.get_data_ptr<element::Type_t::f32>(), 1.0);
    try {
        c.get_data_ptr<element::Type_t::f64>();
        FAIL() << "Bad type not detected.";
    } catch (const AssertFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
    try {
        c.get_data_ptr<element::Type_t::i32>();
        FAIL() << "Bad type not detected.";
    } catch (const AssertFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
}

TEST(constant, bad_get_data_ptr_ov_string) {
    ov::op::v0::Constant c(element::string, Shape{}, vector<std::string>{"abc"});
    EXPECT_EQ(*c.get_data_ptr<element::Type_t::string>(), "abc");
    try {
        c.get_data_ptr<element::Type_t::f64>();
        FAIL() << "Bad type not detected.";
    } catch (const AssertFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
    try {
        c.get_data_ptr<element::Type_t::i32>();
        FAIL() << "Bad type not detected.";
    } catch (const AssertFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("get_data_ptr"));
    }
}

TEST(constant, hold_tensor) {
    Shape shape{4};
    void* hostDataPtr = nullptr;
    std::shared_ptr<ov::op::v0::Constant> constOp;
    {
        auto tensor = ov::Tensor(element::f32, Shape{1, 2, 3, 3});
        hostDataPtr = tensor.data();
        constOp = std::make_shared<ov::op::v0::Constant>(tensor);
    }
    const void* constDataPtr = constOp->get_data_ptr();
    ASSERT_EQ(constDataPtr, hostDataPtr);
    EXPECT_EQ(constOp->get_strides(), Strides({72, 36, 12, 4}));
    EXPECT_EQ(constOp->get_tensor_view().get_strides(), Strides({72, 36, 12, 4}));
}

TEST(constant, hold_tensor_ov_string) {
    Shape shape{4};
    void* hostDataPtr = nullptr;
    std::shared_ptr<ov::op::v0::Constant> constOp;
    {
        auto tensor = ov::Tensor(element::string, Shape{1, 2, 3, 3});
        hostDataPtr = tensor.data();
        constOp = std::make_shared<ov::op::v0::Constant>(tensor);
    }
    const void* constDataPtr = constOp->get_data_ptr();
    ASSERT_EQ(constDataPtr, hostDataPtr);
}

TEST(constant, hold_tensor_custom_strides) {
    const auto shape = Shape{1, 2, 3, 4};
    const auto strides = Strides{192, 96, 32, 8};
    void* shared_data_ptr = nullptr;
    std::shared_ptr<op::v0::Constant> const_op;
    auto storage = std::vector<float>(shape_size(shape));
    {
        auto strided_view = Tensor(element::f32, shape, storage.data(), strides);
        shared_data_ptr = strided_view.data();
        const_op = std::make_shared<op::v0::Constant>(strided_view);
    }

    ASSERT_EQ(const_op->get_data_ptr(), shared_data_ptr);
    EXPECT_EQ(const_op->get_strides(), strides);
    EXPECT_EQ(const_op->get_tensor_view().get_strides(), strides);
}

TEST(constant, hold_tensor_custom_strides_revalidate) {
    const auto shape = Shape{1, 2, 3, 4};
    const auto strides = Strides{192, 96, 32, 8};
    void* shared_data_ptr = nullptr;
    std::shared_ptr<op::v0::Constant> const_op;
    auto storage = std::vector<float>(shape_size(shape));
    {
        auto strided_view = Tensor(element::f32, shape, storage.data(), strides);
        shared_data_ptr = strided_view.data();
        const_op = std::make_shared<op::v0::Constant>(strided_view);
    }

    ASSERT_EQ(const_op->get_data_ptr(), shared_data_ptr);
    EXPECT_EQ(const_op->get_strides(), strides);
    EXPECT_EQ(const_op->get_tensor_view().get_strides(), strides);

    const_op->revalidate_and_infer_types();

    ASSERT_EQ(const_op->get_data_ptr(), shared_data_ptr);
    EXPECT_EQ(const_op->get_strides(), strides);
    EXPECT_EQ(const_op->get_tensor_view().get_strides(), strides);
}

TEST(constant, hold_shared_memory_same_size) {
    auto storage =
        std::make_shared<std::vector<int32_t>>(std::initializer_list<int32_t>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
    {
        auto c = op::v0::Constant(element::i32, Shape{storage->size()}, storage->data(), {});
        std::fill_n(storage->begin() + 3, 4, 0);

        EXPECT_EQ(storage.use_count(), 1);
        EXPECT_EQ(c.get_data_ptr(), storage->data());
        EXPECT_EQ(c.get_vector<int32_t>(), std::vector<int32_t>({1, 2, 3, 0, 0, 0, 0, 4, 3, 2, 1}));
        EXPECT_EQ(c.cast_vector<int32_t>(), std::vector<int32_t>({1, 2, 3, 0, 0, 0, 0, 4, 3, 2, 1}));
    }
    EXPECT_EQ(storage.use_count(), 1);
}

TEST(constant, hold_shared_memory_shape_within_memory_size) {
    auto storage = std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto c = op::v0::Constant(element::u8, Shape{2, 3}, storage.data(), {});

    EXPECT_EQ(c.get_data_ptr(), storage.data());
    EXPECT_EQ(c.get_vector<uint8_t>(), std::vector<uint8_t>({1, 2, 3, 4, 5, 6}));
    EXPECT_EQ(c.cast_vector<uint8_t>(), std::vector<uint8_t>({1, 2, 3, 4, 5, 6}));
}

TEST(constant, hold_shared_memory_different_precision) {
    auto storage = std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1};
    auto c = op::v0::Constant(element::u8, Shape{2, 3, 1}, storage.data(), {});

    EXPECT_EQ(c.get_data_ptr(), storage.data());
    EXPECT_EQ(c.get_vector<uint8_t>(), std::vector<uint8_t>({1, 0, 0, 0, 2, 0}));
    EXPECT_EQ(c.cast_vector<uint8_t>(), std::vector<uint8_t>({1, 0, 0, 0, 2, 0}));
}

TEST(constant, own_shared_memory) {
    struct CustomStorage {
        CustomStorage(std::initializer_list<int16_t> values) : values{std::move(values)} {
            ON_CALL(*this, dtor_impl).WillByDefault(testing::Return());
        }

        ~CustomStorage() {
            dtor_impl();
        }

        MOCK_METHOD(void, dtor_impl, ());

        constexpr ov::element::Type get_element_type() const {
            return ov::element::i16;
        }

        std::vector<int16_t> values{};
    };

    {
        auto storage = std::make_shared<CustomStorage>(std::initializer_list<int16_t>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
        auto c =
            std::make_shared<op::v0::Constant>(storage->get_element_type(), Shape{2}, storage->values.data(), storage);

        EXPECT_EQ(storage.use_count(), 2);

        c = nullptr;
        EXPECT_EQ(storage.use_count(), 1);
        EXPECT_CALL(*storage, dtor_impl).Times(1);
    }

    {
        std::shared_ptr<op::v0::Constant> c;
        CustomStorage* s_ptr;
        {
            auto storage = std::make_shared<testing::StrictMock<CustomStorage>>(
                std::initializer_list<int16_t>{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1});
            s_ptr = storage.get();
            c = std::make_shared<op::v0::Constant>(storage->get_element_type(),
                                                   Shape{2},
                                                   storage->values.data(),
                                                   storage);
        }

        EXPECT_CALL(*s_ptr, dtor_impl).Times(1);
        c = nullptr;
    }
}

// Test verifies 2 things:
// a) Checks that bitwise comparison happens on first call of 'get_all_data_elements_bitwise_identical'
// b) Next call of 'get_all_data_elements_bitwise_identical' takes already calculated value
TEST(constant, lazy_bitwise_identical) {
    auto shape = Shape{10, 1000, 1000};
    auto type = element::i32;
    auto byte_size = shape_size(shape) * sizeof(int32_t);
    auto aligned_weights_buffer = std::make_shared<ov::AlignedBuffer>(byte_size);
    std::memset(aligned_weights_buffer->get_ptr<char>(), 1, byte_size);
    auto weights =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(aligned_weights_buffer->get_ptr<char>(),
                                                                               aligned_weights_buffer->size(),
                                                                               aligned_weights_buffer);

    using namespace std::chrono;
    auto create_constant = [&]() {
        auto constant1 = std::make_shared<op::v0::Constant>(type, shape, weights);
        return constant1;
    };
    const int TIMEOUT_MS = 300;
    size_t created_count = 0;
    {
        auto start = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - start).count() < TIMEOUT_MS) {
            create_constant();  // shall be O(1)
            created_count++;
        }
    }
    size_t bitwise_check_count = 0;
    {
        auto start = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - start).count() < TIMEOUT_MS) {
            auto constant1 = create_constant();
            EXPECT_TRUE(constant1->get_all_data_elements_bitwise_identical());  // can be O(N)
            bitwise_check_count++;
        }
    }

    size_t bitwise_check_count_only = 0;
    auto constant1 = create_constant();
    EXPECT_TRUE(constant1->get_all_data_elements_bitwise_identical());  // first time calculation can be O(N)
    {
        auto start = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - start).count() < TIMEOUT_MS) {
            EXPECT_TRUE(constant1->get_all_data_elements_bitwise_identical());  // next calls shall be O(1)
            bitwise_check_count_only++;
        }
    }
    std::cout << "Created: " << created_count << ", Created+Checked=" << bitwise_check_count
              << ", Checked_cached_value=" << bitwise_check_count_only << "\n";
    // Comparing creation from pre-allocated buffer with creation + checking identical
    // '10' times is guaranteed to be faster here (typical value is ~10'000)
    EXPECT_GT(created_count, bitwise_check_count * 10);

    // Comparing getting comparison value from cache with first-time calculation
    // '10' times is guaranteed to be faster here (typical value is ~200'000)
    EXPECT_GT(bitwise_check_count_only, bitwise_check_count * 10);
}

TEST(constant, cast_vector) {
    std::vector<element::Type_t> types = {
        element::boolean, element::bf16, element::f16, element::f32, element::f64, element::i4,     element::i8,
        element::i16,     element::i32,  element::i64, element::u1,  element::u2,  element::u3,     element::u4,
        element::u6,      element::u8,   element::u16, element::u32, element::u64, element::f4e2m1, element::f8e8m0};
    std::vector<int64_t> data = {0, 1, 0, 0, 1, 1, 0, 1};
    std::vector<int64_t> expected_partial_data = {0, 1, 0, 0, 1, 1};

    for (const auto& type : types) {
        const auto& constant = op::v0::Constant::create(type, Shape{data.size()}, data);

        const auto& default_casted = constant->cast_vector<int64_t>();
        EXPECT_EQ(default_casted, data) << "Constant::cast_vector failed default casting for type " << type;

        int64_t num_elements_for_partial_casting = static_cast<int64_t>(expected_partial_data.size());
        const auto& partially_casted = constant->cast_vector<int64_t>(num_elements_for_partial_casting);
        EXPECT_EQ(partially_casted, expected_partial_data)
            << "Constant::cast_vector failed partial casting for type " << type;

        int64_t num_elements_for_over_casting = static_cast<int64_t>(data.size()) + 10;
        const auto& over_casted = constant->cast_vector<int64_t>(num_elements_for_over_casting);
        EXPECT_EQ(over_casted, data) << "Constant::cast_vector failed for partial casting for type " << type;

        EXPECT_TRUE(constant->cast_vector<int64_t>(0).empty())
            << "Constant::cast_vector failed empty casting for type " << type;
    }
}

TEST(constant, cast_vector_ov_string) {
    element::Type_t type = element::string;
    std::vector<std::string> data = {"a", "b", "c", "d", "e", "f", "g", "h"};
    std::vector<std::string> expected_partial_data = {"a", "b", "c", "d", "e", "f"};

    const auto& constant = op::v0::Constant::create(type, Shape{data.size()}, data);

    const auto& default_casted = constant->cast_vector<std::string>();
    EXPECT_EQ(default_casted, data) << "Constant::cast_vector failed default casting for type " << type;

    int64_t num_elements_for_partial_casting = static_cast<int64_t>(expected_partial_data.size());
    const auto& partially_casted = constant->cast_vector<std::string>(num_elements_for_partial_casting);
    EXPECT_EQ(partially_casted, expected_partial_data)
        << "Constant::cast_vector failed partial casting for type " << type;

    int64_t num_elements_for_over_casting = static_cast<int64_t>(data.size()) + 10;
    const auto& over_casted = constant->cast_vector<std::string>(num_elements_for_over_casting);
    EXPECT_EQ(over_casted, data) << "Constant::cast_vector failed for partial casting for type " << type;

    EXPECT_TRUE(constant->cast_vector<std::string>(0).empty())
        << "Constant::cast_vector failed empty casting for type " << type;
}

TEST(constant, get_values_as) {
    ov::op::v0::Constant c(element::i64, Shape{6}, std::vector<int64_t>{2, -3, 1, 0, 1, 5});

    EXPECT_EQ(c.get_shape_val(), Shape({2, 0, 1, 0, 1, 5}));
    EXPECT_EQ(c.get_strides_val(), Strides({2, 0, 1, 0, 1, 5}));
    EXPECT_EQ(c.get_coordinate_val(), Coordinate({2, 0, 1, 0, 1, 5}));
    EXPECT_EQ(c.get_coordinate_diff_val(), CoordinateDiff({2, 0, 1, 0, 1, 5}));
    EXPECT_EQ(c.get_axis_vector_val(), AxisVector({2, 0, 1, 0, 1, 5}));
    EXPECT_EQ(c.get_axis_set_val(), AxisSet({0, 1, 2, 5}));
}
}  // namespace test
}  // namespace ov

namespace std {

template <>
class numeric_limits<ov::test::TestDType> {
public:
    static constexpr bool is_specialized = true;
    static ov::test::TestDType min() noexcept {
        return numeric_limits<float>::min();
    }
    static ov::test::TestDType max() noexcept {
        return numeric_limits<float>::max();
    }
    static ov::test::TestDType lowest() noexcept {
        return numeric_limits<float>::lowest();
    }

    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
};
}  // namespace std
