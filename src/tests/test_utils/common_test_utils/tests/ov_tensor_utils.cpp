// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

using namespace testing;
using namespace ov::util;

TEST(Comparator, boolean) {
    const bool value = true;
    ov::element::Type element_type = ov::element::boolean;
    ov::Shape shape{1, 4};
    bool values[] = {value, value, value, value};
    bool values_ref[] = {value, value, value, value};
    auto tensor = ov::Tensor(element_type, shape, values);
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, boolean_negative) {
    const bool value = true;
    ov::element::Type element_type = ov::element::boolean;
    ov::Shape shape{1, 4};
    bool values[] = {value, value, value, value};
    bool values_ref[] = {value, value, value, !value};
    auto tensor = ov::Tensor(element_type, shape, values);
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, integer) {
    const int value = 1;
    ov::element::Type element_type = ov::element::i32;
    ov::Shape shape{3, 4};
    std::vector<int> values(ov::shape_size(shape), value);
    std::vector<int> values_ref(ov::shape_size(shape), value);
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref.data());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, integer_negative) {
    const int value = 1;
    ov::element::Type element_type = ov::element::i32;
    ov::Shape shape{3, 4};
    std::vector<int> values(ov::shape_size(shape), value);
    std::vector<int> values_ref(ov::shape_size(shape), value);
    values_ref[ov::shape_size(shape) - 1] = value * 0;
    values_ref[ov::shape_size(shape) / 2] = value * 2;
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref.data());
    ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, float_) {
    const float value = 0.1f;
    ov::element::Type element_type = ov::element::f32;
    ov::Shape shape{3, 4};
    std::vector<float> values(ov::shape_size(shape), value);
    std::vector<float> values_ref(ov::shape_size(shape), value);
    // default rel_threshold * value * 0.5 + abs_threshold to be same
    const auto abs_threshold = std::numeric_limits<float>::epsilon();
    const auto def_threshold = ov::test::utils::get_eps_by_ov_type(element_type) * value * 0.9f + abs_threshold;
    for (auto& value : values) {
        value += def_threshold;
    }
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref.data());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, float_large) {
    const float value = 1e6;
    ov::element::Type element_type = ov::element::f32;
    ov::Shape shape{3, 4};
    std::vector<float> values(ov::shape_size(shape), value);
    std::vector<float> values_ref(ov::shape_size(shape), value);
    // default rel_threshold * value * 0.5 + abs_threshold to be same
    const auto abs_threshold = std::numeric_limits<float>::epsilon();
    const auto def_threshold = ov::test::utils::get_eps_by_ov_type(element_type) * value * 0.99 + abs_threshold;
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] += (i % 2 ? def_threshold : -def_threshold);
    }
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref.data());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, float_negative) {
    const float value = 2.4f;
    ov::element::Type element_type = ov::element::f32;
    ov::Shape shape{3, 4};
    std::vector<float> values(ov::shape_size(shape), value);
    std::vector<float> values_ref(ov::shape_size(shape), value);
    const auto abs_threshold = std::numeric_limits<float>::epsilon();
    const auto def_threshold = ov::test::utils::get_eps_by_ov_type(element_type) * value * 1.1f + abs_threshold;
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] += (i % 2 ? def_threshold : -def_threshold);
    }
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref.data());
    ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, float_extra_small) {
    const float value = 1e-6;
    ov::element::Type element_type = ov::element::f32;
    ov::Shape shape{3, 4};
    std::vector<float> values(ov::shape_size(shape), value);
    std::vector<float> values_ref(ov::shape_size(shape), value);
    const auto abs_threshold = std::numeric_limits<float>::epsilon();
    const auto def_threshold = ov::test::utils::get_eps_by_ov_type(ov::element::f32) * value * 0.8f + abs_threshold;
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] += (i % 2 ? def_threshold : -def_threshold);
    }
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref.data());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, different_shapes) {
    const float value = 1e-1;
    ov::element::Type element_type = ov::element::f32;
    ov::Shape shape{3, 4};
    ov::Shape shape_ref{1, 4};
    std::vector<float> values(ov::shape_size(shape), value);
    std::vector<float> values_ref(ov::shape_size(shape_ref), value);
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type, shape_ref, values_ref.data());
    ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, different_prc_low) {
    const float value = 1e-2;
    ov::element::Type element_type = ov::element::f32;
    ov::element::Type element_type_ref = ov::element::f16;
    ov::Shape shape{3, 4};
    const auto abs_threshold = std::numeric_limits<ov::float16>::epsilon();
    const float threshold = ov::test::utils::get_eps_by_ov_type(element_type_ref) * value * 0.9 + abs_threshold;
    std::vector<float> values(ov::shape_size(shape), value + threshold);
    std::vector<ov::float16> values_ref(ov::shape_size(shape), ov::float16(value));
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type_ref, shape, values_ref.data());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, different_prc_up) {
    const float value = 1e+2;
    ov::element::Type element_type = ov::element::f16;
    ov::element::Type element_type_ref = ov::element::f32;
    ov::Shape shape{3, 4};
    const auto abs_threshold = std::numeric_limits<float>::epsilon();
    const float threshold = ov::test::utils::get_eps_by_ov_type(element_type_ref) * value * 0.9f + abs_threshold;
    float updated_value = value - threshold;
    std::vector<ov::float16> values(ov::shape_size(shape), ov::float16(updated_value));
    std::vector<float> values_ref(ov::shape_size(shape), value);
    auto tensor = ov::Tensor(element_type, shape, values.data());
    auto tensor_ref = ov::Tensor(element_type_ref, shape, values_ref.data());
    OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
}

TEST(Comparator, u4) {
    ov::element::Type element_type = ov::element::u4;
    ov::Shape shape{4};
    // two u4 values are packed per byte, low nibble holds the first (LSB-first) element.
    uint8_t values_ref[] = {0x21, 0x43};  // elements: 1, 2, 3, 4
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0x21, 0x43};  // elements: 1, 2, 3, 4 (identical to reference)
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0x21, 0x4F};  // element[2] changed from 3 to 15
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}

TEST(Comparator, i4) {
    ov::element::Type element_type = ov::element::i4;
    ov::Shape shape{4};
    // signed nibble (two's complement), LSB-first: elements 1, -1, -8, 7
    uint8_t values_ref[] = {0xF1, 0x78};
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0xF1, 0x78};  // identical to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0xF1, 0x18};  // element[3] changed from 7 to 1
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}

TEST(Comparator, u1) {
    ov::element::Type element_type = ov::element::u1;
    ov::Shape shape{8};
    uint8_t values_ref[] = {0xAA};  // 0b10101010
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0xAA};  // identical to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0xAB};  // one bit flipped compared to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}

TEST(Comparator, u2) {
    ov::element::Type element_type = ov::element::u2;
    ov::Shape shape{4};
    uint8_t values_ref[] = {0x1B};  // 0b00011011
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0x1B};  // identical to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0x1F};  // one element differs
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}

TEST(Comparator, u3) {
    ov::element::Type element_type = ov::element::u3;
    ov::Shape shape{8};
    // u3 is a split-bit type: 8 values are packed across 3 bytes.
    uint8_t values_ref[] = {0x12, 0x34, 0x56};
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0x12, 0x34, 0x56};  // identical to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0x12, 0xff, 0x56};
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}

TEST(Comparator, u6) {
    ov::element::Type element_type = ov::element::u6;
    ov::Shape shape{4};
    // u6 is a split-bit type: 4 values are packed across 3 bytes.
    uint8_t values_ref[] = {0x12, 0x34, 0x56};
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0x12, 0x34, 0x56};  // identical to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0x12, 0xff, 0x56};
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}

TEST(Comparator, nf4) {
    ov::element::Type element_type = ov::element::nf4;
    ov::Shape shape{4};
    // nf4 quantization codes, packed same as u4.
    uint8_t values_ref[] = {0x21, 0x43};
    auto tensor_ref = ov::Tensor(element_type, shape, values_ref);
    {
        uint8_t values[] = {0x21, 0x43};  // identical to reference
        auto tensor = ov::Tensor(element_type, shape, values);
        OV_ASSERT_NO_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
    {
        uint8_t values[] = {0x21, 0x4F};  // element[2] code changed from 3 to 15 (very different NF4 level)
        auto tensor = ov::Tensor(element_type, shape, values);
        ASSERT_ANY_THROW(ov::test::utils::compare(tensor_ref, tensor));
    }
}
