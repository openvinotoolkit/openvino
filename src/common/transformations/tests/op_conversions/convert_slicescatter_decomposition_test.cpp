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
namespace {
class ConvertSliceScatterTest : public TransformationTestsF, public testing::WithParamInterface<ov::NodeVector> {
private:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& inputs = GetParam();
        manager.register_pass<ov::pass::ConvertSliceScatter>();
        model = create_v15_model(inputs);
        model_ref = create_decomposed_model(inputs);
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
        comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
        comparator.enable(FunctionsComparator::CmpValues::NAMES);
    }

protected:
    std::shared_ptr<ov::Model> create_v15_model(ov::NodeVector inputs) {
        const auto& data = inputs.at(0);
        const auto& updates = inputs.at(1);
        const auto& start = inputs.at(2);
        const auto& stop = inputs.at(3);
        const auto& step = inputs.at(4);
        ov::ParameterVector params{};
        for (const auto& inp : inputs) {
            const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(inp);
            if (param) {
                params.push_back(param);
            }
        }
        std::shared_ptr<ov::opset15::SliceScatter> slicescatter;
        if (inputs.size() == 5) {
            slicescatter = std::make_shared<ov::opset15::SliceScatter>(data, updates, start, stop, step);
        } else {
            slicescatter = std::make_shared<ov::opset15::SliceScatter>(data, updates, start, stop, step, inputs.at(5));
        }
        slicescatter->set_friendly_name("slicescatter15");
        return std::make_shared<ov::Model>(slicescatter->outputs(), params);
    }

    std::shared_ptr<ov::Model> create_decomposed_model(ov::NodeVector inputs) {
        const auto& data = inputs.at(0);
        const auto& updates = inputs.at(1);
        const auto& start = inputs.at(2);
        const auto& stop = inputs.at(3);
        const auto& step = inputs.at(4);
        ov::ParameterVector params{};
        for (const auto& inp : inputs) {
            const auto& param = ov::as_type_ptr<ov::op::v0::Parameter>(inp);
            if (param) {
                params.push_back(param);
            }
        }
        const auto& const_0 = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        const auto& const_1 = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
        const auto& const_1d_neg_1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        const auto& const_scatter_indices_shape = ov::op::v0::Constant::create(ov::element::i64, {2}, {-1, 1});
        const auto& data_shape = std::make_shared<ov::opset8::ShapeOf>(data, ov::element::i64);
        const auto& num_elements_data = std::make_shared<ov::opset8::ReduceProd>(data_shape, const_0, false);
        const auto& data_indices_flatten =
            std::make_shared<ov::opset8::Range>(const_0, num_elements_data, const_1, ov::element::i64);
        const auto& full_data_indices = std::make_shared<ov::opset8::Reshape>(data_indices_flatten, data_shape, false);
        std::shared_ptr<ov::opset8::Slice> slice_indices;
        if (inputs.size() == 5) {
            slice_indices = std::make_shared<ov::opset8::Slice>(full_data_indices, start, stop, step);
        } else {
            slice_indices = std::make_shared<ov::opset8::Slice>(full_data_indices, start, stop, step, inputs.at(5));
        }
        const auto& slice_indices_flatten =
            std::make_shared<ov::opset8::Reshape>(slice_indices, const_scatter_indices_shape, false);
        const auto& updates_flatten = std::make_shared<ov::opset8::Reshape>(updates, const_1d_neg_1, false);
        const auto& data_flatten = std::make_shared<ov::opset8::Reshape>(data, const_1d_neg_1, false);
        const auto& output_flatten =
            std::make_shared<ov::opset8::ScatterNDUpdate>(data_flatten, slice_indices_flatten, updates_flatten);
        const auto& slicescatter = std::make_shared<ov::opset8::Reshape>(output_flatten, data_shape, false);
        slicescatter->set_friendly_name("slicescatter15");
        return std::make_shared<ov::Model>(slicescatter->outputs(), params);
    }
};

INSTANTIATE_TEST_SUITE_P(
    ConvertSliceScatterDecomposition,
    ConvertSliceScatterTest,
    testing::Values(
        ov::NodeVector{
            std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{256, 10, 15}),
            std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{4, 7, 2}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, -15, 25}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {9, 7, -3}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 1, -1}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {0, 1, -1}),
        },
        ov::NodeVector{
            std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{256, 10, 15}),
            std::make_shared<ov::opset15::Parameter>(ov::element::f32, ov::Shape{4, 7, 2}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, -15, 25}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {9, 7, -3}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 1, -1}),
        },
        ov::NodeVector{
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, -15, 25}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {9, 7, -3}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 1, -1}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {0, 1, -1}),
        },
        ov::NodeVector{
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, -15, 25}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {9, 7, -3}),
            ov::op::v0::Constant::create(ov::element::i32, {3}, {2, 1, -1}),
        },
        ov::NodeVector{
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
        },
        ov::NodeVector{
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
            std::make_shared<ov::opset15::Parameter>(ov::element::i32, ov::PartialShape::dynamic()),
        }));
TEST_P(ConvertSliceScatterTest, CompareFunctions) {}

}  // namespace
