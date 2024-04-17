// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/util/op_types.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/floor_mod.hpp"

using namespace testing;
using namespace ov::util;

using ov::Shape;
using ov::op::v0::Parameter;
using ov::op::v1::Add;
using ov::op::v0::Relu;
using ov::op::v1::ReduceMean;
using ov::op::v1::FloorMod;
using ov::op::v1::Reshape;


TEST(Ranges, ranges_by_type_real) {
    auto p0 = std::make_shared<Parameter>(ov::element::f16, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::f16, Shape{3});
    auto add = std::make_shared<Add>(p0, p1);
    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();

    auto real_range = modelRange.get_general_real_range();

    ov::float16 lowest_tmp = std::numeric_limits<ov::float16>::lowest();
    ov::float16 max_tmp = std::numeric_limits<ov::float16>::max();
    double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
    double max = max_tmp.to_bits();
    double range = max - lowest;
    ASSERT_EQ(real_range->start_from, lowest);
    ASSERT_EQ(real_range->range, range);
    ASSERT_EQ(real_range->resolution, 1);

    for (size_t port = 0; port < add->get_input_size(); ++port) {
        ov::Tensor tensor1 = modelRange.generate_input(add, port, Shape{3});
        auto data1 = tensor1.data<ov::float16>();
        for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
            double value = data1[i];
            ASSERT_GE(value, lowest);
            ASSERT_LE(value, range);
        }
    }
}

TEST(Ranges, ranges_by_type_int) {
    auto p0 = std::make_shared<Parameter>(ov::element::i8, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::i8, Shape{3});
    auto add = std::make_shared<Add>(p0, p1);
    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();

    auto int_range = modelRange.get_general_integral_range();

    ASSERT_EQ(int_range->start_from, std::numeric_limits<int8_t>::lowest());
    uint32_t range = static_cast<double>(std::numeric_limits<int8_t>::max()) - static_cast<double>(std::numeric_limits<int8_t>::lowest());
    ASSERT_EQ(int_range->range, range);
    ASSERT_EQ(int_range->resolution, 1);

    for (size_t port = 0; port < add->get_input_size(); ++port) {
        ov::Tensor tensor1 = modelRange.generate_input(add, port, Shape{3});
        auto data1 = tensor1.data<int8_t>();
        for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
            double value = data1[i];
            ASSERT_GE(value, std::numeric_limits<int8_t>::lowest());
            ASSERT_LE(value, std::numeric_limits<int8_t>::max());
        }
    }
}

TEST(Ranges, intersection_real) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{3});

    auto relu = std::make_shared<Relu>(p0);
    auto add = std::make_shared<Add>(p1, relu);

    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();
    auto real_range = modelRange.get_general_real_range();

    auto relu_range = ov::test::utils::InputGenerateData(-1, 2, 32768);
    ASSERT_EQ(real_range->start_from, relu_range.start_from);
    ASSERT_EQ(real_range->range, relu_range.range);
    ASSERT_EQ(real_range->resolution, relu_range.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(add, 0, Shape{3});
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, relu_range.start_from);
        ASSERT_LE(value, relu_range.range);
    }

    auto defaul_range = ov::test::utils::get_range_by_type(ov::element::Type_t::undefined, testing::internal::Random::kMaxRange);
    auto int_range = modelRange.get_general_integral_range();
    ASSERT_EQ(int_range->start_from, defaul_range.start_from);
    ASSERT_EQ(int_range->range, defaul_range.range);
    ASSERT_EQ(int_range->resolution, defaul_range.resolution);
}

TEST(Ranges, intersection_integral) {
    auto p0 = std::make_shared<Parameter>(ov::element::i32, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::i32, Shape{3});

    auto relu = std::make_shared<Relu>(p0);
    auto add = std::make_shared<Add>(p1, relu);

    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();
    auto int_range = modelRange.get_general_integral_range();

    auto relu_range = ov::test::utils::InputGenerateData(0, 15);
    ASSERT_EQ(int_range->start_from, relu_range.start_from);
    ASSERT_EQ(int_range->range, relu_range.range);
    ASSERT_EQ(int_range->resolution, relu_range.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(add, 0, Shape{3});
    auto data1 = tensor1.data<int32_t>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, relu_range.start_from);
        // 96254
        ASSERT_LE(value, relu_range.range);
    }

    auto defaul_range = ov::test::utils::get_range_by_type(ov::element::Type_t::undefined, testing::internal::Random::kMaxRange);
    auto real_range = modelRange.get_general_real_range();
    ASSERT_EQ(real_range->start_from, defaul_range.start_from);
    ASSERT_EQ(real_range->range, defaul_range.range);
    ASSERT_EQ(real_range->resolution, defaul_range.resolution);
}

TEST(Ranges, spetial_ranges) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2, 3});
    auto p2 = std::make_shared<Parameter>(ov::element::i32, Shape{1});

    auto add = std::make_shared<Add>(p0, p1);
    auto reshape = std::make_shared<Reshape>(add, p2, true);

    auto func = std::make_shared<ov::Model>(reshape, ov::ParameterVector{p0, p1, p2});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();
    auto real_range = modelRange.get_general_real_range();

    auto main_range = ov::test::utils::InputGenerateData(-100, 200, 32768);
    ASSERT_EQ(real_range->start_from, main_range.start_from);
    ASSERT_EQ(real_range->range, main_range.range);
    ASSERT_EQ(real_range->resolution, main_range.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(add, 0, Shape{1, 2, 3});
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, main_range.start_from);
        ASSERT_LE(value, main_range.range);
    }

    auto spetial_range = ov::test::utils::InputGenerateData(0, 256, 1, 1, true);
    ov::Tensor tensor2 = modelRange.generate_input(reshape, 1, Shape{1});
    auto data2 = tensor2.data<int32_t>();
    for (size_t i = 0; i < shape_size(tensor2.get_shape()); ++i) {
        double value = data2[i];
        ASSERT_GE(value, spetial_range.start_from);
        ASSERT_LE(value, spetial_range.range);
    }
}


TEST(Ranges, intersection_range) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p2 = std::make_shared<Parameter>(ov::element::i32, Shape{1});

    auto relu = std::make_shared<Relu>(p0);
    auto add = std::make_shared<Add>(p1, relu);
    auto reduce = std::make_shared<ReduceMean>(add, p2, true);

    auto func = std::make_shared<ov::Model>(reduce, ov::ParameterVector{p0, p1, p2});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();
    auto real_range = modelRange.get_general_real_range();

    auto intersection_range_real = ov::test::utils::InputGenerateData(0, 1, 32768);
    ASSERT_EQ(real_range->start_from, intersection_range_real.start_from);
    ASSERT_EQ(real_range->range, intersection_range_real.range);
    ASSERT_EQ(real_range->resolution, intersection_range_real.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(add, 0, Shape{1});
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, intersection_range_real.start_from);
        ASSERT_LE(value, intersection_range_real.range);
    }

    auto int_range = modelRange.get_general_integral_range();
    auto intersection_range_int = ov::test::utils::InputGenerateData(0, 5, 1000);
    ASSERT_EQ(int_range->start_from, intersection_range_int.start_from);
    ASSERT_EQ(int_range->range, intersection_range_int.range);
    ASSERT_EQ(int_range->resolution, intersection_range_int.resolution);
}

TEST(Ranges, not_intersection) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});

    auto relu = std::make_shared<Relu>(p0);
    auto reduce = std::make_shared<FloorMod>(relu, p1);

    auto func = std::make_shared<ov::Model>(reduce, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.collect_ranges(func, testing::internal::Random::kMaxRange);
    modelRange.find_general_ranges();
    auto real_range = modelRange.get_general_real_range();
    ASSERT_EQ(real_range, nullptr);

    auto int_range = modelRange.get_general_integral_range();
    ASSERT_NE(int_range, nullptr);
}

