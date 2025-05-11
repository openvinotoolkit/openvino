// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_reduce_to_reshape.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/opsets/opset.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

namespace {
Shape get_output_shape(const ov::Shape& input_shape, const std::vector<int32_t>& axis) {
    Shape output_shape;
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
            output_shape.push_back(input_shape.at(i));
        }
    }
    return output_shape;
}
template <typename T>
std::shared_ptr<Model> create_model(const ov::element::Type& element_type,
                                    const ov::Shape& input_shape,
                                    const std::vector<int32_t>& axis) {
    auto input = std::make_shared<op::v0::Parameter>(element_type, input_shape);
    auto axis_const = op::v0::Constant::create(ov::element::i32, Shape{axis.size()}, axis);
    auto reduce = std::make_shared<T>(input, axis_const, false);
    return std::make_shared<Model>(reduce, ParameterVector{input});
}
std::shared_ptr<Model> create_ref_model(const ov::element::Type& element_type,
                                        const ov::Shape& input_shape,
                                        const ov::Shape& output_shape) {
    auto input = std::make_shared<op::v0::Parameter>(element_type, input_shape);
    auto reshape_const = op::v0::Constant::create(element::i64, Shape{output_shape.size()}, output_shape);
    auto reshape = std::make_shared<op::v1::Reshape>(input, reshape_const, false);
    return std::make_shared<Model>(reshape, ParameterVector{input});
}
}  // namespace

TEST(reduce_to_reshape, reduce_max_to_reshape) {
    const Shape input_shape{10, 20, 1, 40};
    const std::vector<int32_t> axis{2};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceMax>(element::f32, input_shape, axis);
    auto model_ref = create_ref_model(element::f32, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_mean_to_reshape) {
    const Shape input_shape{1, 20, 30, 40};
    const std::vector<int32_t> axis{0};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceMean>(element::f32, input_shape, axis);
    auto model_ref = create_ref_model(element::f32, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_min_to_reshape) {
    const Shape input_shape{10, 1, 30, 40};
    const std::vector<int32_t> axis{1};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceMin>(element::f32, input_shape, axis);
    auto model_ref = create_ref_model(element::f32, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_sum_to_reshape) {
    const Shape input_shape{1, 20, 1, 40};
    const std::vector<int32_t> axis{0, 2};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceSum>(element::f32, input_shape, axis);
    auto model_ref = create_ref_model(element::f32, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_sum_to_reshape_fail) {
    const Shape input_shape{1, 20, 30, 40};
    const std::vector<int32_t> axis{0, 2};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceSum>(element::f32, input_shape, axis);
    auto model_ref = model->clone();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_prod_to_reshape) {
    const Shape input_shape{10, 20, 30, 1};
    const std::vector<int32_t> axis{3};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceProd>(element::f32, input_shape, axis);
    auto model_ref = create_ref_model(element::f32, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_logical_and_to_reshape) {
    const Shape input_shape{10, 1, 30, 1};
    const std::vector<int32_t> axis{1, 3};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceLogicalAnd>(element::boolean, input_shape, axis);
    auto model_ref = create_ref_model(element::boolean, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_to_reshape, reduce_logical_or_to_reshape) {
    const Shape input_shape{10, 1, 1, 1};
    const std::vector<int32_t> axis{1, 2, 3};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceLogicalOr>(element::boolean, input_shape, axis);
    auto model_ref = create_ref_model(element::boolean, input_shape, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<pass::ConvertReduceToReshape>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}
