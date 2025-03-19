// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_normalization_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"

using namespace ov;
using namespace opset12;
using namespace element;

namespace {
std::shared_ptr<Model> gen_model(const std::vector<PartialShape>& input_shapes,
                                 element::Type elem_type,
                                 int64_t num_groups,
                                 double eps) {
    const auto data = std::make_shared<Parameter>(elem_type, input_shapes[0]);
    const auto scale = std::make_shared<Parameter>(elem_type, input_shapes[1]);
    const auto bias = std::make_shared<Parameter>(elem_type, input_shapes[2]);

    const auto group_norm = std::make_shared<GroupNormalization>(data, scale, bias, num_groups, eps);

    return std::make_shared<Model>(OutputVector{group_norm->output(0)}, ParameterVector{data, scale, bias});
}

std::shared_ptr<Model> gen_model_ref(const std::vector<PartialShape>& input_shapes,
                                     element::Type elem_type,
                                     int64_t num_groups,
                                     float eps) {
    const auto data = std::make_shared<Parameter>(elem_type, input_shapes[0]);
    const auto scale = std::make_shared<Parameter>(elem_type, input_shapes[1]);
    const auto bias = std::make_shared<Parameter>(elem_type, input_shapes[2]);

    const auto data_shape_node = std::make_shared<ShapeOf>(data, element::i64);
    const auto data_rank_size = data->get_partial_shape().rank().get_length();

    const auto axis_node = Constant::create(element::i64, Shape{}, {0});
    const auto split = std::make_shared<Split>(data_shape_node, axis_node, static_cast<size_t>(data_rank_size));
    auto splits = split->outputs();
    auto num_groups_const = Constant::create(element::i64, Shape{1}, {num_groups});
    // The 4D shape: [N * num_groups, C // num_groups, H, W] is created
    // instead of 5D shape: [N, num_groups, C // num_groups, H, W].
    // The reason is the lack of support for 5D MVN input by some plugins.
    OutputVector new_shape{std::make_shared<Multiply>(splits[0], num_groups_const),
                           std::make_shared<Divide>(splits[1], num_groups_const)};

    for (int64_t i = 2; i < data_rank_size; ++i) {
        new_shape.push_back(splits[i]);
    }

    auto target_shape = std::make_shared<Concat>(new_shape, 0);
    auto data_reshaped = std::make_shared<Reshape>(data, target_shape, true);

    std::vector<int64_t> reduction_axes_val(data_rank_size - 1);
    std::iota(reduction_axes_val.begin(), reduction_axes_val.end(), int64_t(1));
    const auto reduction_axes = Constant::create(element::i64, {reduction_axes_val.size()}, reduction_axes_val);

    auto mvn = std::make_shared<MVN>(data_reshaped, reduction_axes, true, eps, op::MVNEpsMode::INSIDE_SQRT);
    std::shared_ptr<Node> result = std::make_shared<Reshape>(mvn, data_shape_node, true);

    std::vector<int64_t> unsqueeze_axes_val(data_rank_size - 2);
    std::iota(unsqueeze_axes_val.begin(), unsqueeze_axes_val.end(), int64_t(1));
    const auto unsqueeze_axes = Constant::create(element::i64, {unsqueeze_axes_val.size()}, unsqueeze_axes_val);

    result = std::make_shared<Multiply>(result, std::make_shared<Unsqueeze>(scale, unsqueeze_axes));
    result = std::make_shared<Add>(result, std::make_shared<Unsqueeze>(bias, unsqueeze_axes));

    return std::make_shared<Model>(OutputVector{result->output(0)}, ParameterVector{data, scale, bias});
}

}  // namespace

TEST_F(TransformationTestsF, GroupNormalizationDecompositionF32) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecompositionF16) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f16;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_num_groups) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 2;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_eps_cast_to_float_min) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 2;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-39);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, std::numeric_limits<float>::min());

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_eps_cast_to_float_max) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 2;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e+39);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, std::numeric_limits<float>::max());

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_3D) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_5D) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 4, 6, 8}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_data_dynamic_rank_no_decomposition) {
    std::vector<PartialShape> input_shapes{PartialShape::dynamic(), PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    // no decomposition
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_data_rank_2D_no_decomposition) {
    std::vector<PartialShape> input_shapes{PartialShape{2, 12}, PartialShape{12}, PartialShape{12}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    // no decomposition
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_bias_scale_dynamic_rank) {
    std::vector<PartialShape> input_shapes{PartialShape{1, 12, 6, 8}, PartialShape::dynamic(), PartialShape::dynamic()};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, GroupNormalizationDecomposition_dynamic_dims) {
    std::vector<PartialShape> input_shapes{PartialShape{-1, -1, -1, -1}, PartialShape{-1}, PartialShape{-1}};
    const int64_t num_groups = 4;
    element::Type elem_type = element::f32;

    model = gen_model(input_shapes, elem_type, num_groups, 1e-3);
    manager.register_pass<pass::GroupNormalizationDecomposition>();

    model_ref = gen_model_ref(input_shapes, elem_type, num_groups, 1e-3f);

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
