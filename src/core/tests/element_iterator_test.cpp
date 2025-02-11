// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_iterator.hpp"

#include <gmock/gmock.h>

#include <array>

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace test {

using testing::ElementsAre;
using testing::ElementsAreArray;

namespace {
constexpr size_t get_buffer_size(const size_t bit_width, const size_t num_of_elements) {
    return (num_of_elements * bit_width + 7) / 8;
}
}  // namespace

// bits number in comments are counted [b7, b6, ..., b0]
// ---- u1
TEST(ElementIteratorTest, write_u1_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, elements_count>{0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1};
    auto output = std::array<int8_t, get_buffer_size(1, elements_count)>{};
    auto iter = element::iterator<element::u1>(output.data());

    std::copy(input.begin(), input.end(), iter);
    EXPECT_THAT(output, ElementsAre(0x16, 0xB3));
}

TEST(ElementIteratorTest, read_const_u1_data) {
    constexpr auto elements_count = 16;
    constexpr auto input = std::array<int8_t, get_buffer_size(1, elements_count)>{0x21, static_cast<int8_t>(0xa3)};
    auto iter = element::iterator<element::u1>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1));
}

TEST(ElementIteratorTest, read_non_const_u1_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, get_buffer_size(1, elements_count)>{0x21, static_cast<int8_t>(0xa3)};
    auto iter = element::iterator<element::u1>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1));
}

TEST(ElementIteratorTest, read_u1_data_increment_decrement_iterator) {
    auto input = std::array<int8_t, 3>{0x32, static_cast<int8_t>(0xa3), 0x55};
    auto iter = element::iterator<element::u1>(input.data() + 1);

    EXPECT_EQ(*iter--, 1);  // 2nd byte bit7
    EXPECT_EQ(*iter++, 0);  // 1st byte bit0
    EXPECT_EQ(*++iter, 0);  // 2nd byte bit6
    EXPECT_EQ(*iter--, 0);  // 2nd byte bit6
    EXPECT_EQ(*iter, 1);    // 2nd byte bit7
}

TEST(ElementIteratorTest, read_u1_data_iterator_with_offset) {
    auto input = std::array<int8_t, 3>{0x32, static_cast<int8_t>(0xa3), 0x41};
    auto iter = element::iterator<element::u1>(input.data() + 1);

    EXPECT_EQ(*iter, 1);                // 2nd byte bit7
    EXPECT_EQ(*(iter - 2), 1);          // 1st byte bit1
    EXPECT_EQ(*(iter - 5), 1);          // 1st byte bit4
    EXPECT_EQ(*(iter + 1), 0);          // 2nd byte bit6
    EXPECT_EQ(*(iter + 8), 0);          // 3rd byte bit7
    EXPECT_EQ(*(iter + 9), 1);          // 3rd byte bit6
    EXPECT_EQ(*std::prev(iter, 1), 0);  // 1st byte bit0
    EXPECT_EQ(*std::next(iter, 2), 1);  // 2nd byte bit5
}

TEST(ElementIteratorTest, read_u1_from_tensor) {
    auto input = std::array<int8_t, 4>{0x32, static_cast<int8_t>(0xa3), 0x41, 0x11};
    auto t = ov::Tensor(element::u1, Shape{2, 16}, input.data());
    auto iter = element::iterator<element::u1>(static_cast<int8_t*>(t.data(element::u1)));

    EXPECT_THAT(
        std::vector<int8_t>(iter, iter + t.get_size()),
        ElementsAre(0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1));
}

TEST(ElementIteratorTest, u1_value_to_output_stream) {
    constexpr auto value = static_cast<int8_t>(0x80);
    auto iter = element::iterator<element::u1>(&value);

    std::stringstream s;
    s << *iter;

    EXPECT_EQ(s.str(), "1");
}

// ---- u2
TEST(ElementIteratorTest, write_u2_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, elements_count>{2, 0, 1, 3, 0, 0, 3, 3, 1, 2, 1, 2, 3, 2, 1, 0};
    auto output = std::array<int8_t, get_buffer_size(2, elements_count)>{};
    auto iter = element::iterator<element::u2>(output.data());

    std::copy(input.begin(), input.end(), iter);

    EXPECT_THAT(output, ElementsAre(0x87, 0x0f, 0x66, 0xe4));
}

TEST(ElementIteratorTest, read_const_u2_data) {
    constexpr auto elements_count = 16;
    constexpr auto input = std::array<int8_t, get_buffer_size(2, elements_count)>{static_cast<int8_t>(0x87),
                                                                                  0x0f,
                                                                                  0x66,
                                                                                  static_cast<int8_t>(0xe4)};
    auto iter = element::iterator<element::u2>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(2, 0, 1, 3, 0, 0, 3, 3, 1, 2, 1, 2, 3, 2, 1, 0));
}

TEST(ElementIteratorTest, read_non_const_u2_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, get_buffer_size(2, elements_count)>{static_cast<int8_t>(0x87),
                                                                        0x0f,
                                                                        0x66,
                                                                        static_cast<int8_t>(0xe4)};
    auto iter = element::iterator<element::u2>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(2, 0, 1, 3, 0, 0, 3, 3, 1, 2, 1, 2, 3, 2, 1, 0));
}

TEST(ElementIteratorTest, read_u2_data_increment_decrement_iterator) {
    auto input = std::array<int8_t, 2>{0x33, static_cast<int8_t>(0x93)};
    auto iter = element::iterator<element::u2>(input.data() + 1);

    EXPECT_EQ(*iter--, 2);  // 2nd byte 1st half-nibble
    EXPECT_EQ(*iter++, 3);  // 1st byte 4th half-nibble
    EXPECT_EQ(*++iter, 1);  // 2nd byte 2nd half-nibble
    EXPECT_EQ(*iter--, 1);  // 2nd byte 2nd half-nibble
    EXPECT_EQ(*--iter, 3);  // 1st byte 4th half-nibble
}

TEST(ElementIteratorTest, read_u2_data_iterator_with_offset) {
    auto input = std::array<int8_t, 3>{0x43, static_cast<int8_t>(0x93), 0x41};
    auto iter = element::iterator<element::u2>(input.data() + 1);

    EXPECT_EQ(*iter, 2);                // 2nd byte 1st half-nibble
    EXPECT_EQ(*(iter - 3), 0);          // 1st byte 2nd half-nibble
    EXPECT_EQ(*(iter - 4), 1);          // 1st byte 1st half-nibble
    EXPECT_EQ(*(iter + 1), 1);          // 2nd byte 2nd half-nibble
    EXPECT_EQ(*(iter + 7), 1);          // 3rd byte 4th half-nibble
    EXPECT_EQ(*std::prev(iter, 1), 3);  // 1st byte 4th half-nibble
    EXPECT_EQ(*std::next(iter, 2), 0);  // 2nd byte 3rd half-nibble
}

TEST(ElementIteratorTest, u2_value_to_output_stream) {
    constexpr auto value = static_cast<int8_t>(0x80);
    auto iter = element::iterator<element::u2>(&value);

    std::stringstream s;
    s << *iter;

    EXPECT_EQ(s.str(), "2");
}

TEST(ElementIteratorTest, read_u2_from_tensor) {
    auto input = std::array<int8_t, 4>{0x32, static_cast<int8_t>(0xa3), 0x41, 0x11};
    auto t = ov::Tensor(element::u2, Shape{4, 4}, input.data());
    auto iter = element::iterator<element::u2>(static_cast<int8_t*>(t.data(element::u2)));

    EXPECT_THAT(std::vector<int8_t>(iter, iter + t.get_size()),
                ElementsAre(0, 3, 0, 2, 2, 2, 0, 3, 1, 0, 0, 1, 0, 1, 0, 1));
}

// --- u3
TEST(ElementIteratorTest, write_u3_data) {
    constexpr auto elements_count = 8;
    auto input = std::array<int8_t, elements_count>{2, 3, 0, 1, 4, 5, 6, 7};
    auto output = std::array<int8_t, 3>{};
    auto iter = element::iterator<element::u3>(output.data());

    std::copy(input.begin(), input.end(), iter);

    EXPECT_THAT(output, ElementsAre(0b10110001, 0b00011011, 0b00001111));
}

TEST(ElementIteratorTest, read_non_const_u3_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, 6>{0x7a, 0x6f, 0x55, static_cast<int8_t>(0xb1), 0x1b, 0x0f};
    auto iter = element::iterator<element::u3>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(1, 7, 2, 6, 1, 6, 3, 7, 2, 3, 0, 1, 4, 5, 6, 7));
}

TEST(ElementIteratorTest, read_const_u3_data) {
    constexpr auto elements_count = 8;
    constexpr auto input = std::array<int8_t, 3>{static_cast<int8_t>(0b10110001), 0b00011011, 0b00001111};
    auto iter = element::iterator<element::u3>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count), ElementsAre(2, 3, 0, 1, 4, 5, 6, 7));
}

TEST(ElementIteratorTest, read_u3_data_iterator_with_offset) {
    // Has values {1, 7, 2, 6, 1, 6, 3, 7, [2], 3, 0, 1, 4, 5, 6, 7}
    auto input = std::array<int8_t, 6>{0x7a, 0x6f, 0x55, static_cast<int8_t>(0xb1), 0x1b, 0x0f};
    auto iter = element::iterator<element::u3>(input.data() + 3);

    EXPECT_EQ(*iter, 2);
    EXPECT_EQ(*(iter - 3), 6);
    EXPECT_EQ(*(iter - 4), 1);
    EXPECT_EQ(*(iter - 5), 6);
    EXPECT_EQ(*(iter + 1), 3);
    EXPECT_EQ(*(iter + 5), 5);
    EXPECT_EQ(*(iter + 7), 7);
    EXPECT_EQ(*std::prev(iter, 1), 7);
    EXPECT_EQ(*std::next(iter, 2), 0);
}

TEST(ElementIteratorTest, read_u3_from_tensor) {
    // Has values {1, 7, 2, 6, 1, 6, 3, 7, [2], 3, 0, 1, 4, 5, 6, 7}
    auto input = std::array<int8_t, 6>{0x7a, 0x6f, 0x55, static_cast<int8_t>(0xb1), 0x1b, 0x0f};
    auto t = ov::Tensor(element::u3, Shape{4, 2, 2}, input.data());
    auto iter = element::iterator<element::u3>(static_cast<int8_t*>(t.data(element::u3)));

    EXPECT_THAT(std::vector<int8_t>(iter, iter + t.get_size()),
                ElementsAre(1, 7, 2, 6, 1, 6, 3, 7, 2, 3, 0, 1, 4, 5, 6, 7));
}

// --- u4
// nibbles are counted as [n1, n0]
TEST(ElementIteratorTest, write_u4_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, elements_count>{1, 2, 3, 10, 12, 15, 14, 4, 7, 9, 11, 13, 8, 0, 5, 6};
    auto output = std::array<int8_t, get_buffer_size(4, elements_count)>{};
    auto iter = element::iterator<element::u4>(output.data());

    std::copy(input.begin(), input.end(), iter);

    EXPECT_THAT(output, ElementsAre(0x21, 0xa3, 0xfc, 0x4e, 0x97, 0xdb, 0x08, 0x65));
}

TEST(ElementIteratorTest, read_const_u4_data) {
    constexpr auto elements_count = 16;
    constexpr auto byte_size = get_buffer_size(4, elements_count);
    constexpr auto input = std::array<int8_t, byte_size>{0x12,
                                                         0x3a,
                                                         static_cast<int8_t>(0xcf),
                                                         static_cast<int8_t>(0xe4),
                                                         0x79,
                                                         static_cast<int8_t>(0xbd),
                                                         0x08,
                                                         0x56};
    auto iter = element::iterator<element::u4>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(2, 1, 10, 3, 15, 12, 4, 14, 9, 7, 13, 11, 8, 0, 6, 5));
}

TEST(ElementIteratorTest, read_non_const_u4_data) {
    constexpr auto elements_count = 16;
    constexpr auto byte_size = get_buffer_size(4, elements_count);
    auto input = std::array<int8_t, byte_size>{0x12,
                                               0x3a,
                                               static_cast<int8_t>(0xcf),
                                               static_cast<int8_t>(0xe4),
                                               0x79,
                                               static_cast<int8_t>(0xbd),
                                               0x08,
                                               0x56};
    auto iter = element::iterator<element::u4>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(2, 1, 10, 3, 15, 12, 4, 14, 9, 7, 13, 11, 8, 0, 6, 5));
}

TEST(ElementIteratorTest, read_u4_data_increment_decrement_iterator) {
    auto input = std::array<int8_t, 3>{0x12, 0x3a};
    auto iter = element::iterator<element::u4>(input.data() + 1);

    EXPECT_EQ(*iter--, 10);  // 2nd byte 1st nibble
    EXPECT_EQ(*iter++, 1);   // 1st byte 2nd nibble
    EXPECT_EQ(*++iter, 3);   // 2nd byte 2nd nibble
    EXPECT_EQ(*iter--, 3);   // 2nd byte 2nd nibble
    EXPECT_EQ(*--iter, 1);   // 1st byte 2nd nibble
}

TEST(ElementIteratorTest, read_u4_data_iterator_with_offset) {
    auto input = std::array<int8_t, 5>{0x42, 0x3a, 0x61, 0x79, 0x5b};
    auto iter = element::iterator<element::u4>(input.data() + 1);

    EXPECT_EQ(*iter, 10);               // 2nd byte 1st nibble
    EXPECT_EQ(*(iter - 2), 2);          // 1st byte 1st nibble
    EXPECT_EQ(*(iter + 7), 5);          // 5th byte 2nd nibble
    EXPECT_EQ(*(iter + 6), 11);         // 2nd byte 1st nibble
    EXPECT_EQ(*(iter - 1), 4);          // 1st byte 2nd nibble
    EXPECT_EQ(*std::prev(iter, 1), 4);  // 1st byte 2nd nibble
    EXPECT_EQ(*std::next(iter, 2), 1);  // 3rd byte 1st nibble
}

TEST(ElementIteratorTest, read_u4_from_tensor) {
    auto input = std::array<int8_t, 5>{0x42, 0x3a, 0x61, 0x79, 0x5b};
    auto t = ov::Tensor(element::u4, Shape{5, 2}, input.data());
    auto iter = element::iterator<element::u4>(static_cast<int8_t*>(t.data(element::u4)));

    EXPECT_THAT(std::vector<int8_t>(iter, iter + t.get_size()), ElementsAre(2, 4, 10, 3, 1, 6, 9, 7, 11, 5));
}

// --- i4
// nibbles are counted as [n1, n0]
TEST(ElementIteratorTest, write_i4_data) {
    constexpr auto elements_count = 16;
    auto input = std::array<int8_t, elements_count>{1, 2, 3, -6, -4, -1, -2, 4, 7, -7, -5, -3, -8, 0, 5, 6};
    auto output = std::array<int8_t, get_buffer_size(4, elements_count)>{};
    auto iter = element::iterator<element::i4>(output.data());

    std::copy(input.begin(), input.end(), iter);

    EXPECT_THAT(output, ElementsAre(0x21, 0xa3, 0xfc, 0x4e, 0x97, 0xdb, 0x08, 0x65));
}

TEST(ElementIteratorTest, read_const_i4_data) {
    constexpr auto elements_count = 16;
    constexpr auto byte_size = get_buffer_size(4, elements_count);
    constexpr auto input = std::array<int8_t, byte_size>{0x12,
                                                         0x3a,
                                                         static_cast<int8_t>(0xcf),
                                                         static_cast<int8_t>(0xe4),
                                                         0x79,
                                                         static_cast<int8_t>(0xbd),
                                                         0x08,
                                                         0x56};
    auto iter = element::iterator<element::i4>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(2, 1, -6, 3, -1, -4, 4, -2, -7, 7, -3, -5, -8, 0, 6, 5));
}

TEST(ElementIteratorTest, read_non_const_i4_data) {
    constexpr auto elements_count = 16;
    constexpr auto byte_size = get_buffer_size(4, elements_count);
    auto input = std::array<int8_t, byte_size>{0x12,
                                               0x3a,
                                               static_cast<int8_t>(0xcf),
                                               static_cast<int8_t>(0xe4),
                                               0x79,
                                               static_cast<int8_t>(0xbd),
                                               0x08,
                                               0x56};
    auto iter = element::iterator<element::i4>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count),
                ElementsAre(2, 1, -6, 3, -1, -4, 4, -2, -7, 7, -3, -5, -8, 0, 6, 5));
}

TEST(ElementIteratorTest, read_i4_data_increment_decrement_iterator) {
    auto input = std::array<int8_t, 2>{0x12, 0x3a};
    auto iter = element::iterator<element::i4>(input.data() + 1);

    EXPECT_EQ(*iter--, -6);  // 2nd byte 1st nibble
    EXPECT_EQ(*iter++, 1);   // 1st byte 2nd nibble
    EXPECT_EQ(*++iter, 3);   // 2nd byte 2nd nibble
    EXPECT_EQ(*iter--, 3);   // 2nd byte 2nd nibble
    EXPECT_EQ(*--iter, 1);   // 1st byte 2nd nibble
}

TEST(ElementIteratorTest, read_i4_data_iterator_with_offset) {
    auto input = std::array<int8_t, 5>{0x42, 0x3a, 0x61, 0x79, 0x5b};
    auto iter = element::iterator<element::i4>(input.data() + 1);

    EXPECT_EQ(*iter, -6);               // 2nd byte 1st nibble
    EXPECT_EQ(*(iter - 2), 2);          // 1st byte 1st nibble
    EXPECT_EQ(*(iter + 7), 5);          // 5th byte 2nd nibble
    EXPECT_EQ(*(iter + 6), -5);         // 2nd byte 1st nibble
    EXPECT_EQ(*(iter - 1), 4);          // 1st byte 2nd nibble
    EXPECT_EQ(*std::prev(iter, 1), 4);  // 1st byte 2nd nibble
    EXPECT_EQ(*std::next(iter, 2), 1);  // 3rd byte 1st nibble
}

TEST(ElementIteratorTest, i4_value_to_output_stream) {
    constexpr auto value = static_cast<int8_t>(0x19);
    auto iter = element::iterator<element::i4>(&value);

    std::stringstream s;
    s << *iter;

    EXPECT_EQ(s.str(), "-7");
}

TEST(ElementIteratorTest, read_i4_from_tensor) {
    auto input = std::array<int8_t, 5>{0x42, 0x3a, 0x61, 0x79, 0x5b};
    auto t = ov::Tensor(element::i4, Shape{10, 1, 1}, input.data());
    auto iter = element::iterator<element::i4>(static_cast<int8_t*>(t.data(element::i4)));

    EXPECT_THAT(std::vector<int8_t>(iter, iter + t.get_size()), ElementsAre(2, 4, -6, 3, 1, 6, -7, 7, -5, 5));
}

// --- u6
TEST(ElementIteratorTest, write_u6_data) {
    constexpr auto elements_count = 8;
    auto input = std::array<int8_t, elements_count>{2, 1, 0, 3, 18, 49, 35, 16};
    auto output = std::array<int8_t, 6>{};
    auto iter = element::iterator<element::u6>(output.data());

    std::copy(input.begin(), input.end(), iter);

    EXPECT_THAT(output, ElementsAre(0x21, 0x03, 0x00, 0x21, 0x30, 0x79));
}

TEST(ElementIteratorTest, read_non_const_u6_data) {
    constexpr auto elements_count = 8;
    auto input = std::array<int8_t, 6>{0x21, 0x03, 0x00, 0x21, 0x30, 0x79};
    auto iter = element::iterator<element::u6>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count), ElementsAre(2, 1, 0, 3, 18, 49, 35, 16));
}

TEST(ElementIteratorTest, read_const_u6_data) {
    constexpr auto elements_count = 8;
    constexpr auto input = std::array<int8_t, 6>{0x21, 0x03, 0x00, 0x21, 0x30, 0x79};
    auto iter = element::iterator<element::u6>(input.data());

    EXPECT_THAT(std::vector<int8_t>(iter, iter + elements_count), ElementsAre(2, 1, 0, 3, 18, 49, 35, 16));
}

TEST(ElementIteratorTest, read_u6_data_increment_decrement_iterator) {
    // Has values {1, 2, 3, 10, [3], 8, 7, 2}
    auto input = std::array<int8_t, 6>{0x12, 0x3a, 0x00, 0x38, 0x72, 0x00};
    auto iter = element::iterator<element::u6>(input.data() + 3);

    EXPECT_EQ(*iter--, 3);
    EXPECT_EQ(*iter++, 10);
    EXPECT_EQ(*++iter, 8);
    EXPECT_EQ(*iter--, 8);
    EXPECT_EQ(*--iter, 10);
}

TEST(ElementIteratorTest, read_u6_data_iterator_with_offset) {
    // Has values {1, 2, 3, 10, [3], 8, 7, 2, 1, 42, 4, 20}
    auto input = std::array<int8_t, 9>{0x12, 0x3a, 0x00, 0x38, 0x72, 0x00, 0x1a, 0x44, 0x21};
    auto iter = element::iterator<element::u6>(input.data() + 3);

    EXPECT_EQ(*iter, 3);
    EXPECT_EQ(*(iter - 3), 2);
    EXPECT_EQ(*(iter - 4), 1);
    EXPECT_EQ(*(iter - 2), 3);
    EXPECT_EQ(*(iter + 1), 8);
    EXPECT_EQ(*(iter + 5), 42);
    EXPECT_EQ(*(iter + 7), 20);
    EXPECT_EQ(*std::prev(iter, 1), 10);
    EXPECT_EQ(*std::next(iter, 2), 7);
}

TEST(ElementIteratorTest, u6_value_to_output_stream) {
    auto input = std::array<int8_t, 3>{0x12, 0x3a, 0x00};
    auto iter = element::iterator<element::u6>(input.data());

    std::stringstream s;
    s << *iter;

    EXPECT_EQ(s.str(), "1");
}

TEST(ElementIteratorTest, read_u6_from_tensor) {
    // Has values {1, 2, 3, 10, 3, 8, 7, 2, 1, 42, 4, 20}
    auto input = std::array<int8_t, 9>{0x12, 0x3a, 0x00, 0x38, 0x72, 0x00, 0x1a, 0x44, 0x21};
    auto t = ov::Tensor(element::u6, Shape{4, 1, 3}, input.data());
    auto iter = element::iterator<element::u6>(static_cast<int8_t*>(t.data(element::u6)));

    EXPECT_THAT(std::vector<int8_t>(iter, iter + t.get_size()), ElementsAre(1, 2, 3, 10, 3, 8, 7, 2, 1, 42, 4, 20));
}

// --- f4e2m1
// nibbles are counted as [n1, n0]
TEST(ElementIteratorTest, write_f4e2m1_data) {
    constexpr auto elements_count = 16;
    auto input_unpacked =
        std::array<float4_e2m1, elements_count>{-0.5f, 0.0f, 0, 0, 1.0f, 6.0f, -1.0f, -4.0f, 0, 0, 0, 0, 0, 0, 0, 0};
    auto output_packed = std::array<float4_e2m1, get_buffer_size(4, elements_count)>{};
    auto iter = element::iterator<element::f4e2m1>(output_packed.data());

    std::copy_n(input_unpacked.begin(), elements_count, iter);

    EXPECT_EQ(std::vector<float4_e2m1>(iter, iter + elements_count),
              std::vector<float4_e2m1>(
                  {-.5f, 0.0f, 0.0f, 0.0f, 1.0f, 6.0f, -1.0f, -4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}));
}

TEST(ElementIteratorTest, read_const_f4e2m1_data) {
    constexpr auto elements_count = 16;
    constexpr auto byte_size = get_buffer_size(4, elements_count);
    constexpr auto input = std::array<int8_t, byte_size>{0x12,
                                                         0x3a,
                                                         static_cast<int8_t>(0xcf),
                                                         static_cast<int8_t>(0xe4),
                                                         0x79,
                                                         static_cast<int8_t>(0xbd),
                                                         0x08,
                                                         0x56};
    auto iter = element::iterator<element::f4e2m1>(input.data());

    EXPECT_EQ(
        std::vector<float4_e2m1>(iter, iter + elements_count),
        std::vector<float4_e2m1>(
            {1.0f, 0.5f, -1.0f, 1.5f, -6.0f, -2.0f, 2.0f, -4.0f, -0.5f, 6.0f, -3.0f, -1.5f, -0.0f, 0.0f, 4.0f, 3.0f}));
}

TEST(ElementIteratorTest, read_non_const_f4e2m1_data) {
    constexpr auto elements_count = 16;
    constexpr auto byte_size = get_buffer_size(4, elements_count);
    auto input = std::array<int8_t, byte_size>{0x12,
                                               0x3a,
                                               static_cast<int8_t>(0xcf),
                                               static_cast<int8_t>(0xe4),
                                               0x79,
                                               static_cast<int8_t>(0xbd),
                                               0x08,
                                               0x56};
    auto iter = element::iterator<element::f4e2m1>(input.data());

    EXPECT_EQ(
        std::vector<float4_e2m1>(iter, iter + elements_count),
        std::vector<float4_e2m1>(
            {1.0f, 0.5f, -1.0f, 1.5f, -6.0f, -2.0f, 2.0f, -4.0f, -0.5f, 6.0f, -3.0f, -1.5f, -0.0f, 0.0f, 4.0f, 3.0f}));
}

TEST(ElementIteratorTest, read_f4e2m1_data_increment_decrement_iterator) {
    auto input_packed = std::array<int8_t, 2>{0x12, 0x3a};
    auto iter = element::iterator<element::f4e2m1>(input_packed.data() + 1);

    EXPECT_EQ(*iter--, -1.0f);  // 2nd byte 1st nibble
    EXPECT_EQ(*iter++, 0.5f);   // 1st byte 2nd nibble
    EXPECT_EQ(*++iter, 1.5f);   // 2nd byte 2nd nibble
    EXPECT_EQ(*iter--, 1.5f);   // 2nd byte 2nd nibble
    EXPECT_EQ(*--iter, 0.5f);   // 1st byte 2nd nibble
    EXPECT_EQ(*--iter, 1.0f);   // 1st byte 1st nibble
}

TEST(ElementIteratorTest, read_f4e2m1_data_iterator_with_offset) {
    auto input = std::array<int8_t, 5>{0x42, 0x3a, 0x61, 0x79, 0x5b};
    auto iter = element::iterator<element::f4e2m1>(input.data() + 1);

    EXPECT_EQ(*iter, -1.0f);               // 2nd byte 1st nibble
    EXPECT_EQ(*(iter - 2), 1.0f);          // 1st byte 1st nibble
    EXPECT_EQ(*(iter + 7), 3.0f);          // 5th byte 2nd nibble
    EXPECT_EQ(*(iter + 6), -1.5f);         // 5th byte 1st nibble
    EXPECT_EQ(*(iter - 1), 2.0f);          // 1st byte 2nd nibble
    EXPECT_EQ(*std::prev(iter, 1), 2.0f);  // 1st byte 2nd nibble
    EXPECT_EQ(*std::next(iter, 2), 0.5f);  // 3rd byte 1st nibble
}

TEST(ElementIteratorTest, f4e2m1_value_to_output_stream) {
    constexpr auto value = static_cast<int8_t>(0x19);
    auto iter = element::iterator<element::f4e2m1>(&value);

    std::stringstream s;
    s << *iter;

    EXPECT_EQ(s.str(), "-0.5");
}

TEST(ElementIteratorTest, read_f4e2m1_from_tensor) {
    auto input = std::array<int8_t, 5>{0x42, 0x3a, 0x61, 0x79, 0x5b};
    auto t = ov::Tensor(element::f4e2m1, Shape{10, 1, 1}, input.data());
    auto iter = element::iterator<element::f4e2m1>(t.data<float4_e2m1>());

    EXPECT_EQ(std::vector<float4_e2m1>(iter, iter + t.get_size()),
              std::vector<float4_e2m1>({1.0f, 2.0f, -1.0f, 1.5f, 0.5f, 4.0f, -0.5f, 6.0f, -1.5f, 3.0f}));
}

}  // namespace test
}  // namespace ov
