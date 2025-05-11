// Copyright (C) 2023 Intel Corporation
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
