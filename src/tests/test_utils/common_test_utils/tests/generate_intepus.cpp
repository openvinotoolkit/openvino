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
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/floor_mod.hpp"

using namespace testing;
using namespace ov::util;

using ov::Shape;
using ov::op::v0::Constant;
using ov::op::v0::Convert;
using ov::op::v0::Parameter;
using ov::op::v1::Add;
using ov::op::v0::Relu;
using ov::op::v1::ReduceMean;
using ov::op::v1::FloorMod;
using ov::op::v1::Reshape;


TEST(Ranges, ranges_by_type) {
    auto p0 = std::make_shared<Parameter>(ov::element::f16, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::i8, Shape{3});

    auto convert = std::make_shared<Convert>(p1, ov::element::f16);
    auto add = std::make_shared<Add>(p0, convert);

    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    auto ranges = ov::test::utils::collect_ranges(func, testing::internal::Random::kMaxRange);

    ASSERT_EQ(ranges.size(), 2);

    std::string range_id = ov::test::utils::get_range_id(add, 0);
    ov::float16 lowest_tmp = std::numeric_limits<ov::float16>::lowest();
    ov::float16 max_tmp = std::numeric_limits<ov::float16>::max();
    double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
    double max = max_tmp.to_bits();
    double range = max - lowest;
    ASSERT_EQ(ranges[range_id]->start_from, lowest);
    ASSERT_EQ(ranges[range_id]->range, range);
    ASSERT_EQ(ranges[range_id]->resolution, 1);

    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(add->get_type_info());
    ov::Tensor tensor1 = it->second(add, 0, add->input(0).get_element_type(), Shape{3}, ranges[range_id]);
    auto data1 = tensor1.data<ov::float16>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, lowest);
        ASSERT_LE(value, range);
    }

    range_id = ov::test::utils::get_range_id(convert, 0);
    ASSERT_EQ(ranges[range_id]->start_from, std::numeric_limits<int8_t>::lowest());
    range = static_cast<double>(std::numeric_limits<int8_t>::max()) - static_cast<double>(std::numeric_limits<int8_t>::lowest());
    ASSERT_EQ(ranges[range_id]->range, range);
    ASSERT_EQ(ranges[range_id]->resolution, 1);

    it = inputMap.find(convert->get_type_info());
    ov::Tensor tensor2 = it->second(convert, 0, convert->input(0).get_element_type(), Shape{3}, ranges[range_id]);
    auto data2 = tensor2.data<int8_t>();
    for (size_t i = 0; i < shape_size(tensor2.get_shape()); ++i) {
        double value = data2[i];
        ASSERT_GE(value, std::numeric_limits<int8_t>::lowest());
        ASSERT_LE(value, std::numeric_limits<int8_t>::max());
    }
}

TEST(Ranges, intersection_real) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{3});

    auto relu = std::make_shared<Relu>(p0);
    auto add = std::make_shared<Add>(p1, relu);

    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    auto ranges = ov::test::utils::collect_ranges(func, testing::internal::Random::kMaxRange);

    auto defaul_range = ov::test::utils::get_range_by_type(ov::element::Type_t::undefined, testing::internal::Random::kMaxRange);
    auto relu_range = ov::test::utils::InputGenerateData(-1, 2, 32768);

    ASSERT_EQ(ranges.size(), 2);
    ASSERT_EQ(ranges["1"]->start_from, relu_range.start_from);
    ASSERT_EQ(ranges["1"]->range, relu_range.range);
    ASSERT_EQ(ranges["1"]->resolution, relu_range.resolution);

    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(add->get_type_info());
    ov::Tensor tensor1 = it->second(add, 0, add->input(0).get_element_type(), Shape{3}, ranges["1"]);
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, relu_range.start_from);
        // 96254
        ASSERT_LE(value, relu_range.range);
    }

    ASSERT_EQ(ranges["0"]->start_from, defaul_range.start_from);
    ASSERT_EQ(ranges["0"]->range, defaul_range.range);
    ASSERT_EQ(ranges["0"]->resolution, defaul_range.resolution);
}

TEST(Ranges, intersection_integral) {
    auto p0 = std::make_shared<Parameter>(ov::element::i32, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::i32, Shape{3});

    auto relu = std::make_shared<Relu>(p0);
    auto add = std::make_shared<Add>(p1, relu);

    auto func = std::make_shared<ov::Model>(add, ov::ParameterVector{p0, p1});

    auto ranges = ov::test::utils::collect_ranges(func, testing::internal::Random::kMaxRange);

    auto defaul_range = ov::test::utils::get_range_by_type(ov::element::Type_t::undefined, testing::internal::Random::kMaxRange);
    auto relu_range = ov::test::utils::InputGenerateData(0, 15);

    ASSERT_EQ(ranges.size(), 2);
    ASSERT_EQ(ranges["0"]->start_from, relu_range.start_from);
    ASSERT_EQ(ranges["0"]->range, relu_range.range);
    ASSERT_EQ(ranges["0"]->resolution, relu_range.resolution);

    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(add->get_type_info());
    ov::Tensor tensor1 = it->second(add, 0, add->input(0).get_element_type(), Shape{3}, ranges["0"]);
    auto data1 = tensor1.data<int32_t>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, relu_range.start_from);
        // 96254
        ASSERT_LE(value, relu_range.range);
    }

    ASSERT_EQ(ranges["1"]->start_from, defaul_range.start_from);
    ASSERT_EQ(ranges["1"]->range, defaul_range.range);
    ASSERT_EQ(ranges["1"]->resolution, defaul_range.resolution);
}


TEST(Ranges, spetial_ranges) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2, 3});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2, 3});
    auto p2 = std::make_shared<Parameter>(ov::element::i32, Shape{1});

    auto add = std::make_shared<Add>(p0, p1);
    auto reshape = std::make_shared<Reshape>(add, p2, true);

    auto func = std::make_shared<ov::Model>(reshape, ov::ParameterVector{p0, p1, p2});

    auto ranges = ov::test::utils::collect_ranges(func, testing::internal::Random::kMaxRange);

    ASSERT_EQ(ranges.size(), 3);
    auto main_range = ov::test::utils::InputGenerateData(-100, 200, 32768);
    ASSERT_EQ(ranges["1"]->start_from, main_range.start_from);
    ASSERT_EQ(ranges["1"]->range, main_range.range);
    ASSERT_EQ(ranges["1"]->resolution, main_range.resolution);

    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(add->get_type_info());
    ov::Tensor tensor1 = it->second(add, 0, add->input(0).get_element_type(), Shape{1, 2, 3}, ranges["1"]);
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, main_range.start_from);
        ASSERT_LE(value, main_range.range);
    }

    auto spetial_range = ov::test::utils::InputGenerateData(0, 256, 1, 1, true);
    std::string range_id = ov::test::utils::get_range_id(reshape, 1, true);
    ASSERT_EQ(ranges[range_id]->start_from, spetial_range.start_from);
    ASSERT_EQ(ranges[range_id]->range, spetial_range.range);
    ASSERT_EQ(ranges[range_id]->resolution, spetial_range.resolution);

    it = inputMap.find(reshape->get_type_info());
    ov::Tensor tensor2 = it->second(reshape, 1, reshape->input(1).get_element_type(), Shape{1}, ranges[range_id]);
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

    auto ranges = ov::test::utils::collect_ranges(func, testing::internal::Random::kMaxRange);

    // wrong resolution
    auto intersection_range_real = ov::test::utils::InputGenerateData(0, 1, 1000);

    ASSERT_EQ(ranges.size(), 2);
    ASSERT_EQ(ranges["1"]->start_from, intersection_range_real.start_from);
    ASSERT_EQ(ranges["1"]->range, intersection_range_real.range);
    ASSERT_EQ(ranges["1"]->resolution, intersection_range_real.resolution);

    auto inputMap = ov::test::utils::getInputMap();
    auto it = inputMap.find(add->get_type_info());
    ov::Tensor tensor1 = it->second(add, 0, add->input(0).get_element_type(), Shape{3}, ranges["1"]);
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, intersection_range_real.start_from);
        ASSERT_LE(value, intersection_range_real.range);
    }

    auto intersection_range_int = ov::test::utils::InputGenerateData(0, 5);
    ASSERT_EQ(ranges["0"]->start_from, intersection_range_int.start_from);
    ASSERT_EQ(ranges["0"]->range, intersection_range_int.range);
    ASSERT_EQ(ranges["0"]->resolution, intersection_range_int.resolution);
}

TEST(Ranges, not_intersection) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});

    auto relu = std::make_shared<Relu>(p0);
    auto reduce = std::make_shared<FloorMod>(relu, p1);

    auto func = std::make_shared<ov::Model>(reduce, ov::ParameterVector{p0, p1});

    auto ranges = ov::test::utils::collect_ranges(func, testing::internal::Random::kMaxRange);

    ASSERT_EQ(ranges.size(), 0);
}

