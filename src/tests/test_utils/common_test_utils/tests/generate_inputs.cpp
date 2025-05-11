// Copyright (C) 20234 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/util/op_types.hpp"
#include "common_test_utils/type_ranges.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/reshape.hpp"

using namespace testing;
using namespace ov::util;

using ov::Shape;
using ov::op::v0::Parameter;
using ov::op::v0::Result;
using ov::op::v0::Concat;
using ov::op::v0::Relu;
using ov::op::v1::ReduceMean;
using ov::op::v1::FloorMod;
using ov::op::v1::Reshape;

TEST(RangesTests, ranges_by_type_real) {
    auto p0 = std::make_shared<Parameter>(ov::element::f16, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::f16, Shape{3});
    auto concat = std::make_shared<Concat>(ov::OutputVector{p0, p1}, 0);
    auto func = std::make_shared<ov::Model>(concat, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);

    auto real_range = modelRange.get_range_for_param(p0);

    ov::float16 lowest_tmp = std::numeric_limits<ov::float16>::lowest();
    ov::float16 max_tmp = std::numeric_limits<ov::float16>::max();
    double lowest = 0 - static_cast<double>(lowest_tmp.to_bits());
    double max = max_tmp.to_bits();
    double range = max - lowest;
    ASSERT_EQ(real_range->start_from, lowest);
    ASSERT_EQ(real_range->range, range);
    ASSERT_EQ(real_range->resolution, 1);

    for (size_t port = 0; port < concat->get_input_size(); ++port) {
        ov::Tensor tensor1 = modelRange.generate_input(concat, port, Shape{3});
        auto data1 = tensor1.data<ov::float16>();
        for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
            double value = data1[i];
            ASSERT_GE(value, lowest);
            ASSERT_LE(value, range);
        }
    }
}

TEST(RangesTests, ranges_by_type_int) {
    auto p0 = std::make_shared<Parameter>(ov::element::i8, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::i8, Shape{3});
    auto concat = std::make_shared<Concat>(ov::OutputVector{p0, p1}, 0);
    auto func = std::make_shared<ov::Model>(concat, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);

    auto int_range = modelRange.get_range_for_param(p0);

    ASSERT_EQ(int_range->start_from, std::numeric_limits<int8_t>::lowest());
    uint32_t range = static_cast<double>(std::numeric_limits<int8_t>::max()) - static_cast<double>(std::numeric_limits<int8_t>::lowest());
    ASSERT_EQ(int_range->range, range);
    ASSERT_EQ(int_range->resolution, 1);

    for (size_t port = 0; port < concat->get_input_size(); ++port) {
        ov::Tensor tensor1 = modelRange.generate_input(concat, port, Shape{3});
        auto data1 = tensor1.data<int8_t>();
        for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
            double value = data1[i];
            ASSERT_GE(value, std::numeric_limits<int8_t>::lowest());
            ASSERT_LE(value, std::numeric_limits<int8_t>::max());
        }
    }
}

TEST(RangesTests, intersection_real) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{3});

    auto relu = std::make_shared<Relu>(p0);
    auto concat = std::make_shared<Concat>(ov::OutputVector{p1, relu}, 0);

    auto func = std::make_shared<ov::Model>(concat, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);
    auto relu_range = modelRange.get_range_for_param(p0);

    auto relu_range_ref = ov::test::utils::InputGenerateData(-1, 2, 32768);
    ASSERT_EQ(relu_range->start_from, relu_range_ref.start_from);
    ASSERT_EQ(relu_range->range, relu_range_ref.range);
    ASSERT_EQ(relu_range->resolution, relu_range_ref.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(relu, 0, Shape{3});
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, relu_range_ref.start_from);
        ASSERT_LE(value, relu_range_ref.range);
    }

    auto concat_range_ref = ov::test::utils::rangeByType.get_range(ov::element::f32);
    auto concat_range = modelRange.get_range_for_param(p1);
    ASSERT_EQ(concat_range->start_from, concat_range_ref.start_from);
    ASSERT_EQ(concat_range->range, concat_range_ref.range);
    ASSERT_EQ(concat_range->resolution, concat_range_ref.resolution);

    ov::Tensor tensor2 = modelRange.generate_input(concat, 0, Shape{3});
    auto data2 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor2.get_shape()); ++i) {
        double value = data2[i];
        ASSERT_GE(value, concat_range_ref.start_from);
        ASSERT_LE(value, concat_range_ref.range);
    }
}

TEST(RangesTests, intersection_integral) {
    auto p0 = std::make_shared<Parameter>(ov::element::i32, Shape{3});
    auto p1 = std::make_shared<Parameter>(ov::element::i32, Shape{3});

    auto relu = std::make_shared<Relu>(p0);
    auto concat = std::make_shared<Concat>(ov::OutputVector{p1, relu}, 0);

    auto func = std::make_shared<ov::Model>(concat, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);
    auto relu_range = modelRange.get_range_for_param(p0);

    auto relu_range_ref = ov::test::utils::InputGenerateData(0, 15);
    ASSERT_EQ(relu_range->start_from, relu_range_ref.start_from);
    ASSERT_EQ(relu_range->range, relu_range_ref.range);
    ASSERT_EQ(relu_range->resolution, relu_range_ref.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(relu, 0, Shape{3});
    auto data1 = tensor1.data<int32_t>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, relu_range_ref.start_from);
        ASSERT_LE(value, relu_range_ref.range);
    }

    auto concat_range_ref = ov::test::utils::rangeByType.get_range(ov::element::f32);
    auto concat_range = modelRange.get_range_for_param(p1);
    ASSERT_EQ(concat_range->start_from, concat_range_ref.start_from);
    ASSERT_EQ(concat_range->range, concat_range_ref.range);
    ASSERT_EQ(concat_range->resolution, concat_range_ref.resolution);

    ov::Tensor tensor2 = modelRange.generate_input(concat, 0, Shape{3});
    auto data2 = tensor1.data<int32_t>();
    for (size_t i = 0; i < shape_size(tensor2.get_shape()); ++i) {
        double value = data2[i];
        ASSERT_GE(value, concat_range_ref.start_from);
        ASSERT_LE(value, concat_range_ref.range);
    }
}

TEST(RangesTests, spetial_ranges) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2, 3});
    p0->set_friendly_name("p0");
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2, 3});
    p1->set_friendly_name("p1");
    auto p2 = std::make_shared<Parameter>(ov::element::i32, Shape{1});
    p2->set_friendly_name("p2");

    auto concat = std::make_shared<Concat>(ov::OutputVector{p0, p1}, 1);
    concat->set_friendly_name("Concat");
    auto reshape = std::make_shared<Reshape>(concat, p2, true);
    reshape->set_friendly_name("reshape");

    auto res = std::make_shared<Result>(reshape);

    auto func = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{p0, p1, p2});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);
    auto real_range = modelRange.get_range_for_param(p0);

    auto main_range = ov::test::utils::InputGenerateData(-100, 200, 32768);
    ASSERT_EQ(real_range->start_from, main_range.start_from);
    ASSERT_EQ(real_range->range, main_range.range);
    ASSERT_EQ(real_range->resolution, main_range.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(concat, 0, Shape{1, 2, 3});
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, main_range.start_from);
        ASSERT_LE(value, main_range.range);
    }

    auto spetial_range_ref = ov::test::utils::InputGenerateData(0, 256, 1, 1, true);
    auto spetial_range = modelRange.get_range_for_param(p2);
    ASSERT_EQ(spetial_range->start_from, spetial_range_ref.start_from);
    ASSERT_EQ(spetial_range->range, spetial_range_ref.range);
    ASSERT_EQ(spetial_range->resolution, spetial_range_ref.resolution);

    ov::Tensor tensor2 = modelRange.generate_input(reshape, 1, Shape{1});
    auto data2 = tensor2.data<int32_t>();
    for (size_t i = 0; i < shape_size(tensor2.get_shape()); ++i) {
        double value = data2[i];
        ASSERT_GE(value, spetial_range_ref.start_from);
        ASSERT_LE(value, spetial_range_ref.range);
    }
}

TEST(RangesTests, intersection_range) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p2 = std::make_shared<Parameter>(ov::element::i32, Shape{1});

    auto relu = std::make_shared<Relu>(p0);
    auto concat = std::make_shared<Concat>(ov::OutputVector{p1, relu}, 1);
    auto reduce = std::make_shared<ReduceMean>(concat, p2, true);

    auto func = std::make_shared<ov::Model>(reduce, ov::ParameterVector{p0, p1, p2});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);
    auto real_range = modelRange.get_range_for_param(p0);

    auto intersection_range_real = ov::test::utils::InputGenerateData(0, 1, 32768);
    ASSERT_EQ(real_range->start_from, intersection_range_real.start_from);
    ASSERT_EQ(real_range->range, intersection_range_real.range);
    ASSERT_EQ(real_range->resolution, intersection_range_real.resolution);

    ov::Tensor tensor1 = modelRange.generate_input(relu, 0, Shape{1});
    auto data1 = tensor1.data<float>();
    for (size_t i = 0; i < shape_size(tensor1.get_shape()); ++i) {
        double value = data1[i];
        ASSERT_GE(value, intersection_range_real.start_from);
        ASSERT_LE(value, intersection_range_real.range);
    }

    auto int_range = modelRange.get_range_for_param(p2);
    auto intersection_range_int = ov::test::utils::InputGenerateData(0, 5, 1000);
    ASSERT_EQ(int_range->start_from, intersection_range_int.start_from);
    ASSERT_EQ(int_range->range, intersection_range_int.range);
    ASSERT_EQ(int_range->resolution, intersection_range_int.resolution);
}

TEST(RangesTests, not_intersection) {
    auto p0 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});
    auto p1 = std::make_shared<Parameter>(ov::element::f32, Shape{1, 2});

    auto relu = std::make_shared<Relu>(p0);
    auto floorMod = std::make_shared<FloorMod>(relu, p1);

    auto func = std::make_shared<ov::Model>(floorMod, ov::ParameterVector{p0, p1});

    ov::test::utils::ModelRange modelRange;
    modelRange.find_mode_ranges(func);

    auto not_intersection_range = modelRange.get_range_for_param(p0);
    auto not_intersection_range_ref = ov::test::utils::InputGenerateData(-1, 2, 32768);
    ASSERT_EQ(not_intersection_range->start_from, not_intersection_range_ref.start_from);
    ASSERT_EQ(not_intersection_range->range, not_intersection_range_ref.range);
    ASSERT_EQ(not_intersection_range->resolution, not_intersection_range_ref.resolution);

    auto floorMod_range = modelRange.get_range_for_param(p1);
    auto floorMod_range_ref = ov::test::utils::InputGenerateData(2, 2, 128);
    ASSERT_EQ(floorMod_range->start_from, floorMod_range_ref.start_from);
    ASSERT_EQ(floorMod_range->range, floorMod_range_ref.range);
    ASSERT_EQ(floorMod_range->resolution, floorMod_range_ref.resolution);
}

