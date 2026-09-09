// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <plugin/transformations/decompose_one_hot_non_const_values.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {

using NegativeIndicesMode = ov::op::v16::OneHot::NegativeIndicesMode;

std::shared_ptr<ov::op::v0::Constant> bool_const(bool value) {
    return std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, value);
}

// on_value taken from the shape of a dynamic tensor, as in the ONNX export of `x.repeat(1, N)`.
ov::Output<ov::Node> shape_scalar(const ov::Output<ov::Node>& data) {
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i32);
    auto axes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    return std::make_shared<ov::op::v0::Squeeze>(shape_of, axes);
}

}  // namespace

TEST_F(TransformationTestsF, DecomposeOneHotNonConstOnValueV16) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
        auto on = shape_scalar(data);
        auto indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {2});
        auto off = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        auto one_hot =
            std::make_shared<ov::op::v16::OneHot>(indices, depth, on, off, 1, NegativeIndicesMode::NORMALIZE);

        model = std::make_shared<ov::Model>(ov::OutputVector{one_hot}, ov::ParameterVector{data});
        manager.register_pass<DecomposeOneHotNonConstValues>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
        auto on = shape_scalar(data);
        auto indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {2});
        auto off = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        auto mask = std::make_shared<ov::op::v16::OneHot>(indices,
                                                          depth,
                                                          bool_const(true),
                                                          bool_const(false),
                                                          1,
                                                          NegativeIndicesMode::NORMALIZE);
        auto select = std::make_shared<ov::op::v1::Select>(mask, on, off);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, DecomposeOneHotNonConstOffValueV1) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto off = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        auto on = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {5.0f});
        auto one_hot = std::make_shared<ov::op::v1::OneHot>(indices, depth, on, off, 1);

        model = std::make_shared<ov::Model>(ov::OutputVector{one_hot}, ov::ParameterVector{indices, off});
        manager.register_pass<DecomposeOneHotNonConstValues>();
    }
    {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto off = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        auto on = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {5.0f});
        auto mask = std::make_shared<ov::op::v1::OneHot>(indices, depth, bool_const(true), bool_const(false), 1);
        auto select = std::make_shared<ov::op::v1::Select>(mask, on, off);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{indices, off});
    }
}

TEST_F(TransformationTestsF, DecomposeOneHotNonConstOnAndOffValuesV16IgnoreNegative) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, -1});
        auto on = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto off = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {3});
        auto one_hot = std::make_shared<ov::op::v16::OneHot>(indices,
                                                             depth,
                                                             on,
                                                             off,
                                                             2,
                                                             NegativeIndicesMode::IGNORE_NEGATIVE);

        model = std::make_shared<ov::Model>(ov::OutputVector{one_hot}, ov::ParameterVector{indices, on, off});
        manager.register_pass<DecomposeOneHotNonConstValues>();
    }
    {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, -1});
        auto on = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto off = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {3});
        auto mask = std::make_shared<ov::op::v16::OneHot>(indices,
                                                          depth,
                                                          bool_const(true),
                                                          bool_const(false),
                                                          2,
                                                          NegativeIndicesMode::IGNORE_NEGATIVE);
        auto select = std::make_shared<ov::op::v1::Select>(mask, on, off);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{select}, ov::ParameterVector{indices, on, off});
    }
}

// Constant on/off are handled by the one_hot primitive itself, the pass must leave them alone.
TEST_F(TransformationTestsF, DecomposeOneHotConstValuesIsNoop) {
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto depth = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {3});
        auto on = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
        auto off = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
        auto one_hot = std::make_shared<ov::op::v1::OneHot>(indices, depth, on, off, 1);

        model = std::make_shared<ov::Model>(ov::OutputVector{one_hot}, ov::ParameterVector{indices});
        manager.register_pass<DecomposeOneHotNonConstValues>();
    }
    // model_ref is intentionally left unset: the model must stay as it is.
}
