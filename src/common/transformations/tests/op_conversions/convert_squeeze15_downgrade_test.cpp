// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_squeeze15_downgrade.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset15.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

namespace {

enum class IndicesMode { NONE, CONST, PARAM };

std::shared_ptr<ov::Model> create_v15_model(const IndicesMode indices_mode,
                                            const std::vector<int> indices_const_val,
                                            const bool allow_axis_skip) {
    const PartialShape data_shape{-1, {2, 5}, 1, {1, 5}, 4};
    const auto& data = std::make_shared<ov::opset15::Parameter>(ov::element::f32, data_shape);
    ov::ParameterVector params = {data};
    std::shared_ptr<op::v15::Squeeze> squeeze;
    if (indices_mode == IndicesMode::NONE) {
        squeeze = std::make_shared<ov::opset15::Squeeze>(data, allow_axis_skip);
    } else if (indices_mode == IndicesMode::PARAM) {
        const auto& indices =
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, PartialShape({data_shape.rank()}));
        params.push_back(indices);
        squeeze = std::make_shared<ov::opset15::Squeeze>(data, indices, allow_axis_skip);
    } else if (indices_mode == IndicesMode::CONST) {
        const auto& indices =
            ov::opset15::Constant::create(ov::element::i32, Shape({indices_const_val.size()}), indices_const_val);
        squeeze = std::make_shared<ov::opset15::Squeeze>(data, indices, allow_axis_skip);
    }
    squeeze->set_friendly_name("squeeze15");
    return std::make_shared<ov::Model>(squeeze->outputs(), params);
}

std::shared_ptr<ov::Model> create_v1_model(const IndicesMode indices_mode, const std::vector<int> indices_const_val) {
    const PartialShape data_shape{-1, {2, 5}, 1, {1, 5}, 4};
    const auto& data = std::make_shared<ov::opset1::Parameter>(ov::element::f32, data_shape);
    ov::ParameterVector params = {data};
    std::shared_ptr<op::v0::Squeeze> squeeze;
    if (indices_mode == IndicesMode::NONE) {
        squeeze = std::make_shared<ov::opset1::Squeeze>(data);
    } else if (indices_mode == IndicesMode::PARAM) {
        const auto& indices =
            std::make_shared<ov::opset1::Parameter>(ov::element::i32, PartialShape({data_shape.rank()}));
        params.push_back(indices);
        squeeze = std::make_shared<ov::opset1::Squeeze>(data, indices);
    } else if (indices_mode == IndicesMode::CONST) {
        const auto& indices =
            ov::opset1::Constant::create(ov::element::i32, Shape({indices_const_val.size()}), indices_const_val);
        squeeze = std::make_shared<ov::opset1::Squeeze>(data, indices);
    }
    squeeze->set_friendly_name("squeeze15");
    return std::make_shared<ov::Model>(squeeze->outputs(), params);
}

}  // namespace

TEST_F(TransformationTestsF, ConvertSqueeze15ToSqueeze1_no_indices_no_skip) {
    manager.register_pass<ov::pass::ConvertSqueeze15ToSqueeze0>();
    model = create_v15_model(IndicesMode::NONE, {}, false);
    model_ref = create_v1_model(IndicesMode::NONE, {});
    EXPECT_EQ(model->output(0).get_partial_shape(), model_ref->output(0).get_partial_shape());
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

TEST_F(TransformationTestsF, ConvertSqueeze15ToSqueeze1_no_indices_skip) {
    manager.register_pass<ov::pass::ConvertSqueeze15ToSqueeze0>();
    model = create_v15_model(IndicesMode::NONE, {}, true);
    model_ref = create_v1_model(IndicesMode::NONE, {});
    EXPECT_EQ(model->output(0).get_partial_shape(), model_ref->output(0).get_partial_shape());
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

TEST_F(TransformationTestsF, ConvertSqueeze15ToSqueeze1_const_indices_no_skip) {
    manager.register_pass<ov::pass::ConvertSqueeze15ToSqueeze0>();
    model = create_v15_model(IndicesMode::CONST, {0, -4, 3}, false);
    model_ref = create_v1_model(IndicesMode::CONST, {0, -4, 3});
    EXPECT_EQ(model->output(0).get_partial_shape(), model_ref->output(0).get_partial_shape());
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

TEST_F(TransformationTestsF, ConvertSqueeze15ToSqueeze1_dynamic_indices_no_skip) {
    manager.register_pass<ov::pass::ConvertSqueeze15ToSqueeze0>();
    model = create_v15_model(IndicesMode::PARAM, {}, false);
    model_ref = create_v1_model(IndicesMode::PARAM, {});
    EXPECT_EQ(model->output(0).get_partial_shape(), model_ref->output(0).get_partial_shape());
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::NAMES);
}

TEST_F(TransformationTestsF, ConvertSqueeze15ToSqueeze1_unsupported_skip) {
    manager.register_pass<ov::pass::ConvertSqueeze15ToSqueeze0>();
    model = create_v15_model(IndicesMode::PARAM, {}, true);
}
