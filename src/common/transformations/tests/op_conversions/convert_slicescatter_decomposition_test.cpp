// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/convert_slicescatter.hpp"
#include "transformations/utils/utils.hpp"
using namespace testing;

namespace {

std::shared_ptr<ov::Model> create_v15_model(bool with_axes) {
    const auto data = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{256, 10, 15});
    const auto updates = std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{4, 7, 2});
    const auto start = ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 0, 0});
    const auto stop = ov::op::v0::Constant::create(ov::element::i32, {3}, {9, 7, 2});
    const auto step = ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 1, 1});
    const auto axes = ov::op::v0::Constant::create(ov::element::i32, {3}, {0, 1, 2});
    std::shared_ptr<ov::opset15::SliceScatter> slicescatter;
    if (!with_axes) {
        slicescatter = std::make_shared<ov::opset15::SliceScatter>(data, updates, start, stop, step);
    } else {
        slicescatter = std::make_shared<ov::opset15::SliceScatter>(data, updates, start, stop, step, axes);
    }
    slicescatter->set_friendly_name("slicescatter15");
    return std::make_shared<ov::Model>(slicescatter->outputs(), ov::ParameterVector{data, updates});
}

std::shared_ptr<ov::Model> create_decomposed_model(bool with_axes) {
    const auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{256, 10, 15});
    const auto updates = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{4, 7, 2});
    const auto start = ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 0, 0});
    const auto stop = ov::op::v0::Constant::create(ov::element::i32, {3}, {9, 7, 2});
    const auto step = ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 1, 1});
    const auto axes = ov::op::v0::Constant::create(ov::element::i32, {3}, {0, 1, 2});
    auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto neg_one_1d = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto scatter_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1});
    auto data_shape = std::make_shared<ov::opset8::ShapeOf>(data, ov::element::i64);
    auto num_elements_data = std::make_shared<ov::opset8::ReduceProd>(data_shape, zero, false);
    auto data_indices_flattened = std::make_shared<ov::opset8::Range>(zero, num_elements_data, one, ov::element::i64);
    auto full_data_indices = std::make_shared<ov::opset8::Reshape>(data_indices_flattened, data_shape, false);
    std::shared_ptr<ov::opset8::Slice> slice_indices;
    if (!with_axes) {
        slice_indices = std::make_shared<ov::opset8::Slice>(full_data_indices, start, stop, step);
    } else {
        slice_indices = std::make_shared<ov::opset8::Slice>(full_data_indices, start, stop, step, axes);
    }
    auto slice_indices_flatten = std::make_shared<ov::opset8::Reshape>(slice_indices, scatter_shape, false);
    auto updates_flatten = std::make_shared<ov::opset8::Reshape>(updates, neg_one_1d, false);
    auto data_flatten = std::make_shared<ov::opset8::Reshape>(data, neg_one_1d, false);
    auto output_flatten =
        std::make_shared<ov::opset8::ScatterNDUpdate>(data_flatten, slice_indices_flatten, updates_flatten);
    auto slicescatter = std::make_shared<ov::opset8::Reshape>(output_flatten, data_shape, false);
    slicescatter->set_friendly_name("slicescatter15");

    return std::make_shared<ov::Model>(slicescatter->outputs(), ov::ParameterVector{data, updates});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertSliceScatter15Decomposition_axes) {
    manager.register_pass<ov::pass::ConvertSliceScatter>();
    model = create_v15_model(true);
    model_ref = create_decomposed_model(true);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}

TEST_F(TransformationTestsF, ConvertSliceScatter15Decomposition_no_axes) {
    manager.register_pass<ov::pass::ConvertSliceScatter>();
    model = create_v15_model(false);
    model_ref = create_decomposed_model(false);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
