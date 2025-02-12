// // Copyright (C) 2018-2025 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

#include "transformations/op_conversions/einsum_decomposition.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov;
namespace {
using namespace ov::gen_pattern;
std::shared_ptr<ov::Node> extract_subshape_from_shape(const std::shared_ptr<ov::Node>& shape_node,
                                                      size_t begin,
                                                      size_t end) {
    auto const_begin = makeConst(element::i64, ov::Shape({1}), {begin});
    auto const_end = makeConst(element::i64, ov::Shape({1}), {end});
    auto const_1 = makeConst(element::i64, ov::Shape({1}), {1});
    auto subshape = makeOP<opset1::StridedSlice>({shape_node, const_begin, const_end, const_1},
                                                 {{"begin_mask", {0}},
                                                  {"end_mask", {0}},
                                                  {"new_axis_mask", {}},
                                                  {"shrink_axis_mask", {}},
                                                  {"ellipsis_mask", {}}});
    return subshape;
}

std::shared_ptr<ov::Node> broadcast_merge_shapes(const std::shared_ptr<ov::Node>& shape_node_lhs,
                                                 const std::shared_ptr<ov::Node>& shape_node_rhs) {
    auto const_1 = makeConst(element::i64, ov::Shape({1}), {1});
    auto tensor_of_lhs_shape = makeOP<opset3::Broadcast>({const_1, shape_node_lhs}, {{"mode", "numpy"}});
    auto tensor_of_broadcasted_lhs_rhs_shape =
        makeOP<opset3::Broadcast>({tensor_of_lhs_shape, shape_node_rhs}, {{"mode", "bidirectional"}});
    auto broadcasted_shapes = makeOP<opset3::ShapeOf>({tensor_of_broadcasted_lhs_rhs_shape}, {{"output_type", "i64"}});
    return broadcasted_shapes;
}

std::shared_ptr<ov::Node> create_identity(const std::shared_ptr<ov::Node>& data,
                                          const std::vector<size_t>& repated_label_indices) {
    auto shapeof_data = makeOP<opset3::ShapeOf>({data}, {{"output_type", "i64"}});
    auto rankof_data = makeOP<opset1::ShapeOf>({shapeof_data});
    auto const_0 = makeConst(element::i64, ov::Shape({}), {0});
    auto const_1 = makeConst(element::i64, ov::Shape({}), {1});
    auto num_of_repeated_labels = makeConst(element::i64, ov::Shape({}), {repated_label_indices.size()});
    auto repeated_label_indices = makeConst(element::i64,
                                            ov::Shape({
                                                repated_label_indices.size(),
                                            }),
                                            repated_label_indices);
    auto repeated_dimensions =
        makeOP<opset7::Gather>({shapeof_data, repeated_label_indices, const_0}, {{"batch_dims", 0}});
    auto repeated_dimensions_size = makeOP<opset1::ReduceProd>({repeated_dimensions, const_0}, {{"keep_dims", true}});
    auto zeros_of_size = makeOP<opset1::Broadcast>({const_0, repeated_dimensions_size}, {{"mode", "numpy"}});
    auto repeated_dimension = makeOP<opset7::Gather>({repeated_dimensions, const_0, const_0}, {{"batch_dims", 0}});
    auto range_max_val =
        makeOP<opset1::Power>({repeated_dimension, num_of_repeated_labels}, {{"auto_broadcast", "numpy"}});
    auto step_numerator = makeOP<opset1::Subtract>({range_max_val, const_1}, {{"auto_broadcast", "numpy"}});
    auto step_numerator_but_not_0 = makeOP<opset1::Maximum>({step_numerator, const_1}, {{"auto_broadcast", "numpy"}});
    auto step_denominator = makeOP<opset1::Subtract>({repeated_dimension, const_1}, {{"auto_broadcast", "numpy"}});
    auto step_denominator_but_not_0 =
        makeOP<opset1::Maximum>({step_denominator, const_1}, {{"auto_broadcast", "numpy"}});
    auto step = makeOP<opset1::Divide>({step_numerator_but_not_0, step_denominator_but_not_0},
                                       {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
    auto eye_flattened_indices = makeOP<opset1::Range>({const_0, range_max_val, step});
    auto repeated_dimension_1d = makeOP<opset1::Unsqueeze>({repeated_dimension, const_0});
    auto ones = makeOP<opset1::Broadcast>({const_1, repeated_dimension_1d}, {{"mode", "numpy"}});
    auto eye_flattened = makeOP<opset3::ScatterElementsUpdate>({zeros_of_size, eye_flattened_indices, ones, const_0});
    auto ones_of_input_shape_rank = makeOP<opset1::Broadcast>({const_1, rankof_data}, {{"mode", "numpy"}});
    auto identity_shape = makeOP<opset3::ScatterElementsUpdate>(
        {ones_of_input_shape_rank, repeated_label_indices, repeated_dimensions, const_0});
    auto identity = makeOP<opset1::Reshape>({eye_flattened, identity_shape}, {{"special_zero", false}});
    return identity;
}

std::shared_ptr<ov::Node> extract_diagonal(const std::shared_ptr<ov::Node>& data,
                                           const std::vector<std::vector<size_t>>& indices_of_repeated_labels) {
    // Initialize multi_identity by identity for first repeated label.
    auto multi_identity = create_identity(data, indices_of_repeated_labels[0]);
    // Initialize reduction axes by all except first repated_label_indices for first repeated label.
    std::vector<size_t> reduce_axes(indices_of_repeated_labels[0].begin() + 1, indices_of_repeated_labels[0].end());
    // Merge remaining identities.
    for (size_t i = 1; i < indices_of_repeated_labels.size(); i++) {
        auto identity = create_identity(data, indices_of_repeated_labels[i]);
        multi_identity = makeOP<opset1::Multiply>({multi_identity, identity}, {{"auto_broadcast", "numpy"}});
        reduce_axes.insert(reduce_axes.end(),
                           indices_of_repeated_labels[i].begin() + 1,
                           indices_of_repeated_labels[i].end());
    }
    // Convert to match type of data
    auto multi_identity_cvt = makeOP<opset1::ConvertLike>({multi_identity, data});
    auto unreduced_diagonal = makeOP<opset1::Multiply>({data, multi_identity_cvt}, {{"auto_broadcast", "numpy"}});
    auto const_reduce_axes = makeConst(element::i64, ov::Shape({reduce_axes.size()}), reduce_axes);
    auto diagonal = makeOP<opset1::ReduceSum>({unreduced_diagonal, const_reduce_axes}, {{"keep_dims", false}});
    return diagonal;
}

}  // namespace
TEST_F(TransformationTestsF, Einsum_2in_matmul) {
    PartialShape data_shape_1{5, 2};
    PartialShape data_shape_2{10, 1, 25};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto einsum = std::make_shared<opset7::Einsum>(OutputVector{data_1, data_2}, "kl,mlj->mkj");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::EinsumDecomposition>();
    }
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);

        // Transpose data_2 so that common labels, separated and reduced labels are grouped for both operands.
        auto order_2 = ov::op::v0::Constant::create(element::i64, {3}, {0, 2, 1});
        auto transpose_2 = std::make_shared<ov::op::v1::Transpose>(data_2, order_2);

        // Broadcast data_1 and data_2 to common broadcasted shapes for common and reduced subshapes.
        // Subgraphes are constant-folded, target subshapes are calculated broadcast_merge_shapes function.
        auto broadcast_shape_constant_1 =
            ov::op::v0::Constant::create(element::i64, Shape{data_shape_1.size()}, {5, 2});
        auto broadcast_shape_constant_2 =
            ov::op::v0::Constant::create(element::i64, Shape{data_shape_2.size()}, {10, 25, 2});
        auto broadcast_1 = std::make_shared<ov::op::v3::Broadcast>(data_1,
                                                                   broadcast_shape_constant_1,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        auto broadcast_2 = std::make_shared<ov::op::v3::Broadcast>(transpose_2,
                                                                   broadcast_shape_constant_2,
                                                                   ov::op::BroadcastType::BIDIRECTIONAL);
        // Optionally reshape broadcasted data_1 and data_2 so separate and reduced labels are represented by one
        // dimension. Subgraphes are constant-folded, target subshapes are calculated broadcast_merge_shapes function.
        auto shape_constant_1 = ov::op::v0::Constant::create(element::i64, Shape{2}, {5, 2});
        auto shape_constant_2 = ov::op::v0::Constant::create(element::i64, Shape{2}, {250, 2});
        auto reshape_1 = std::make_shared<ov::op::v1::Reshape>(broadcast_1, shape_constant_1, false);
        auto reshape_2 = std::make_shared<ov::op::v1::Reshape>(broadcast_2, shape_constant_2, false);
        // Apply MatMul operation for formatted inputs.
        auto matmul = std::make_shared<ov::op::v0::MatMul>(reshape_1, reshape_2, false, true);
        // Optionally reshape back by unrolling dimensions corresponding to separate labels if needed.
        // Subgraphes are constant-folded, target subshapes are calculated broadcast_merge_shapes function.
        auto shape_out = ov::op::v0::Constant::create(element::i64, {3}, {5, 10, 25});
        auto reshape_out = std::make_shared<ov::op::v1::Reshape>(matmul, shape_out, false);
        // Transpose to the original order of output labels.
        auto order_out = ov::op::v0::Constant::create(element::i64, {3}, {1, 0, 2});
        auto transpose_out = std::make_shared<ov::op::v1::Transpose>(reshape_out, order_out);

        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, Einsum_2in_matmul_dynamic) {
    PartialShape data_shape_1 = PartialShape::dynamic(2);
    PartialShape data_shape_2 = PartialShape::dynamic(3);
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto einsum = std::make_shared<opset7::Einsum>(OutputVector{data_1, data_2}, "kl,mlj->mkj");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::EinsumDecomposition>();
    }
    {
        using namespace ov::gen_pattern;
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        // Transpose data_2 so that common labels, separated and reduced labels are grouped for both operands.
        auto Constant_485 = makeConst(element::i64,
                                      ov::Shape({
                                          3,
                                      }),
                                      {0, 2, 1});
        auto Transpose_486 = makeOP<opset1::Transpose>({data_2, Constant_485});
        // Get shapes of data_1 and data_2.
        auto ShapeOf_data_1 = makeOP<opset3::ShapeOf>({data_1}, {{"output_type", "i64"}});
        auto ShapeOf_data_2 = makeOP<opset3::ShapeOf>({Transpose_486}, {{"output_type", "i64"}});

        // Get reduced subshape for data_1.
        auto reduced1 = extract_subshape_from_shape(ShapeOf_data_1, 1, 2);

        // Get reduced subshape for data_2.
        auto reduced2 = extract_subshape_from_shape(ShapeOf_data_2, 2, 3);

        // broadcast_merge_shapes(reduced1, reduced2)
        auto reduced_subshape_broadcast_merge_shapes = broadcast_merge_shapes(reduced1, reduced2);

        // Extract separate subshape for data_1.
        auto separate1_subshape = extract_subshape_from_shape(ShapeOf_data_1, 0, 1);

        // Extract separate subshape for data_2.
        auto separate2_subshape = extract_subshape_from_shape(ShapeOf_data_2, 0, 2);

        // Broadcast data_1 and data_2 based on caluculated subshapes.
        auto Concat_512 =
            makeOP<opset1::Concat>({separate1_subshape, reduced_subshape_broadcast_merge_shapes}, {{"axis", 0}});
        auto Broadcast_data_1 = makeOP<opset3::Broadcast>({data_1, Concat_512}, {{"mode", "bidirectional"}});
        auto Concat_521 =
            makeOP<opset1::Concat>({separate2_subshape, reduced_subshape_broadcast_merge_shapes}, {{"axis", 0}});
        auto Broadcast_data_2 = makeOP<opset3::Broadcast>({Transpose_486, Concat_521}, {{"mode", "bidirectional"}});

        // Optionally reshape broadcasted data_1 and data_2 so separate and reduced labels are represented by one
        // dimension. Subgraphes are constant-folded, target subshapes are calculated broadcast_merge_shapes function.
        // Reshape 1
        auto Constant_525 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {0});
        // Reduce separate and reduced
        auto Separate1_subshape_red =
            makeOP<opset1::ReduceProd>({separate1_subshape, Constant_525}, {{"keep_dims", true}});
        auto reduced1_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        // Merge subshapes
        auto reshape_subshape1 = makeOP<opset1::Concat>({Separate1_subshape_red, reduced1_subshape_red}, {{"axis", 0}});
        auto Reshape_1 = makeOP<opset1::Reshape>({Broadcast_data_1, reshape_subshape1}, {{"special_zero", false}});
        // Reshape 2
        auto Constant_569 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {0});
        // Reduce separate and reduced
        auto Separate2_subshape_red =
            makeOP<opset1::ReduceProd>({separate2_subshape, Constant_569}, {{"keep_dims", true}});
        auto Reduced2_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        // Merge subshapes
        auto reshape_subshape2 = makeOP<opset1::Concat>({Separate2_subshape_red, Reduced2_subshape_red}, {{"axis", 0}});
        auto Reshape_2 = makeOP<opset1::Reshape>({Broadcast_data_2, reshape_subshape2}, {{"special_zero", false}});

        // Apply MatMul operation for formatted inputs.
        auto matmul = std::make_shared<ov::op::v0::MatMul>(Reshape_1, Reshape_2, false, true);

        // Optionally reshape back by unrolling dimensions corresponding to separate labels if needed.
        // Target subshapes are calculated broadcast_merge_shapes function and concatenated.
        auto shape_out = makeOP<opset1::Concat>({separate1_subshape, separate2_subshape}, {{"axis", 0}});
        auto reshape_out = std::make_shared<ov::op::v1::Reshape>(matmul, shape_out, false);
        // Transpose to the original order of output labels.
        auto order_out = ov::op::v0::Constant::create(element::i64, {3}, {1, 0, 2});
        auto transpose_out = std::make_shared<ov::op::v1::Transpose>(reshape_out, order_out);

        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, Einsum_2in_matmul_ellipsis_dynamic) {
    PartialShape data_shape_1 = PartialShape::dynamic(2);
    PartialShape data_shape_2 = PartialShape::dynamic(5);
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto einsum = std::make_shared<opset7::Einsum>(OutputVector{data_1, data_2}, "kl...,m...lj->mkj");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1, data_2});
        manager.register_pass<ov::pass::EinsumDecomposition>();
    }
    {
        using namespace ov::gen_pattern;
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        // Process data_1
        // data_1 contains no dimensions at ellipsis label, unsqueeze to allow for broadcasting
        auto Constant_1200 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Unsqueeze_1201 = makeOP<opset1::Unsqueeze>({data_1, Constant_1200});
        // Match ranks of dimensions covered by ellipsis labels
        auto Constant_1202 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto data_1_processed = makeOP<opset1::Unsqueeze>({Unsqueeze_1201, Constant_1202});
        // Process data_2
        // Transpose data_2 so that common labels, separated and reduced labels are grouped for both operands.
        auto Constant_1204 = makeConst(element::i64,
                                       ov::Shape({
                                           5,
                                       }),
                                       {0, 4, 3, 1, 2});
        auto data_2_processed = makeOP<opset1::Transpose>({data_2, Constant_1204});

        // Get shapes for data_1 and data_2
        auto ShapeOf_data_1 = makeOP<opset3::ShapeOf>({data_1_processed}, {{"output_type", "i64"}});
        auto ShapeOf_data_2 = makeOP<opset3::ShapeOf>({data_2_processed}, {{"output_type", "i64"}});

        // Get reduced subshape for data_1.
        auto reduced1 = extract_subshape_from_shape(ShapeOf_data_1, 1, 4);

        // Get reduced subshape for data_2.
        auto reduced2 = extract_subshape_from_shape(ShapeOf_data_2, 2, 5);

        // broadcast_merge_shapes(reduced1, reduced_2)
        auto reduced_subshape_broadcast_merge_shapes = broadcast_merge_shapes(reduced1, reduced2);

        // Extract separate subshape for data_1.
        auto separate1_subshape = extract_subshape_from_shape(ShapeOf_data_1, 0, 1);

        // Extract separate subshape for data_2.
        auto separate2_subshape = extract_subshape_from_shape(ShapeOf_data_2, 0, 2);

        // Broadcast data_1 and data_2 based on caluculated subshapes.
        auto Concat_1231 =
            makeOP<opset1::Concat>({separate1_subshape, reduced_subshape_broadcast_merge_shapes}, {{"axis", 0}});
        auto Broadcast_data_1 = makeOP<opset3::Broadcast>({data_1_processed, Concat_1231}, {{"mode", "bidirectional"}});
        auto Concat_1240 =
            makeOP<opset1::Concat>({separate2_subshape, reduced_subshape_broadcast_merge_shapes}, {{"axis", 0}});
        auto Broadcast_data_2 = makeOP<opset3::Broadcast>({data_2_processed, Concat_1240}, {{"mode", "bidirectional"}});

        // Optionally reshape broadcasted data_1 and data_2 so separate and reduced labels are represented by one
        // dimension. Subgraphes are constant-folded, target subshapes are calculated broadcast_merge_shapes function.
        // Reshape 1
        auto Constant_1244 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Separate1_subshape_red =
            makeOP<opset1::ReduceProd>({separate1_subshape, Constant_1244}, {{"keep_dims", true}});
        auto reduced1_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        auto reshape1_shape = makeOP<opset1::Concat>({Separate1_subshape_red, reduced1_subshape_red}, {{"axis", 0}});
        auto Reshape_1 = makeOP<opset1::Reshape>({Broadcast_data_1, reshape1_shape}, {{"special_zero", false}});

        // Reshape 2
        auto Constant_1302 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Separate2_subshape_red =
            makeOP<opset1::ReduceProd>({separate2_subshape, Constant_1302}, {{"keep_dims", true}});
        auto Reduced2_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        auto reshape2_shape = makeOP<opset1::Concat>({Separate2_subshape_red, Reduced2_subshape_red}, {{"axis", 0}});
        auto Reshape_2 = makeOP<opset1::Reshape>({Broadcast_data_2, reshape2_shape}, {{"special_zero", false}});
        // Apply MatMul operation for formatted inputs.
        auto matmul = makeOP<opset1::MatMul>({Reshape_1, Reshape_2}, {{"transpose_a", false}, {"transpose_b", true}});
        // Optionally reshape back by unrolling dimensions corresponding to separate labels if needed.
        // Target subshapes are calculated broadcast_merge_shapes function and concatenated.
        auto reshape_outshape = makeOP<opset1::Concat>({separate1_subshape, separate2_subshape}, {{"axis", 0}});
        auto reshape_out = makeOP<opset1::Reshape>({matmul, reshape_outshape}, {{"special_zero", false}});
        // Transpose to the original order of output labels.
        auto Constant_1363 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {1, 0, 2});
        auto transpose_out = makeOP<opset1::Transpose>({reshape_out, Constant_1363});
        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1, data_2});
    }
}

TEST_F(TransformationTestsF, Einsum_1in_repeated_labels_ellipsis_static_cf) {
    Shape data_shape_1 = {1, 3, 2, 1, 3, 1};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto einsum = std::make_shared<opset7::Einsum>(OutputVector{data_1}, "ij...iji->j...i");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1});
        manager.register_pass<ov::pass::EinsumDecomposition>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }
    {
        using namespace ov::gen_pattern;
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        // If shapes are static, multi-identity can be constant-folded.
        auto multi_identity = makeConst(
            element::f32,
            ov::Shape({
                1,
                3,
                1,
                1,
                3,
                1,
            }),
            {1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f});
        auto Multiply_1383 = makeOP<opset1::Multiply>({data_1, multi_identity}, {{"auto_broadcast", "numpy"}});
        auto Constant_1384 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {3, 4, 5});
        auto data_1_diagonal = makeOP<opset1::ReduceSum>({Multiply_1383, Constant_1384}, {{"keep_dims", false}});
        // Transpose to the original order of output labels.
        auto Constant_1386 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {1, 2, 0});
        auto transpose_out = makeOP<opset1::Transpose>({data_1_diagonal, Constant_1386});
        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1});
    }
}

TEST_F(TransformationTestsF, Einsum_1in_repeated_labels_empty_ellipsis_dynamic) {
    PartialShape data_shape_1 = PartialShape::dynamic(5);
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto einsum = std::make_shared<opset7::Einsum>(OutputVector{data_1}, "ij...iji->j...i");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1});
        manager.register_pass<ov::pass::EinsumDecomposition>();
    }
    {
        using namespace ov::gen_pattern;
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);

        // Extract diagonal
        auto data_1_diagonal = extract_diagonal(data_1,
                                                {
                                                    {0, 2, 4},  // indices of repeated label i
                                                    {1, 3},     // indices of repeated label j
                                                });

        // Transpose to the original order of output labels.
        auto Constant_3027 = makeConst(element::i64,
                                       ov::Shape({
                                           2,
                                       }),
                                       {1, 0});
        auto transpose_out = makeOP<opset1::Transpose>({data_1_diagonal, Constant_3027});
        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1});
    }
}

TEST_F(TransformationTestsF, Einsum_3in_broadcast_duplicated_ellipsis_repeated_static_cf) {
    PartialShape data_shape_1 = {1, 2, 2, 1, 1, 1};
    PartialShape data_shape_2 = {4, 1, 1, 1, 1, 1};
    PartialShape data_shape_3 = {3, 1, 3, 3};
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto data_3 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_3);
        auto einsum =
            std::make_shared<opset7::Einsum>(OutputVector{data_1, data_2, data_3}, "ba...b,bcccdd,...dbcc->c...b");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1, data_2, data_3});
        manager.register_pass<ov::pass::EinsumDecomposition>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }
    {
        using namespace ov::gen_pattern;
        auto node_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_3);
        auto node_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto node_4 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto Multiply_1990 = makeConst(element::f32,
                                       ov::Shape({
                                           1,
                                           1,
                                           1,
                                           1,
                                           1,
                                           1,
                                       }),
                                       {1.000000f});
        auto Multiply_1991 = makeOP<opset1::Multiply>({node_2, Multiply_1990}, {{"auto_broadcast", "numpy"}});
        auto Constant_1992 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {2, 3, 5});
        auto ReduceSum_1993 = makeOP<opset1::ReduceSum>({Multiply_1991, Constant_1992}, {{"keep_dims", false}});
        auto Concat_2034 = makeConst(element::i64,
                                     ov::Shape({
                                         3,
                                     }),
                                     {4, 3, 3});
        auto Broadcast_2035 = makeOP<opset3::Broadcast>({ReduceSum_1993, Concat_2034}, {{"mode", "bidirectional"}});
        auto Concat_2051 = makeConst(element::i64,
                                     ov::Shape({
                                         4,
                                     }),
                                     {4, 3, 3, 1});
        auto Reshape_2052 = makeOP<opset1::Reshape>({Broadcast_2035, Concat_2051}, {{"special_zero", false}});
        auto Convert_1700 = makeConst(element::f32,
                                      ov::Shape({
                                          1,
                                          1,
                                          1,
                                          1,
                                          1,
                                          1,
                                      }),
                                      {1.000000f});
        auto Multiply_1701 = makeOP<opset1::Multiply>({node_4, Convert_1700}, {{"auto_broadcast", "numpy"}});
        auto Constant_1702 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {5});
        auto ReduceSum_1703 = makeOP<opset1::ReduceSum>({Multiply_1701, Constant_1702}, {{"keep_dims", false}});
        auto Constant_1799 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto ReduceSum_1800 = makeOP<opset1::ReduceSum>({ReduceSum_1703, Constant_1799}, {{"keep_dims", false}});
        auto Constant_1803 = makeConst(element::i64,
                                       ov::Shape({
                                           2,
                                       }),
                                       {4, 5});
        auto Unsqueeze_1804 = makeOP<opset1::Unsqueeze>({ReduceSum_1800, Constant_1803});
        auto Constant_1605 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Unsqueeze_1606 = makeOP<opset1::Unsqueeze>({node_0, Constant_1605});
        auto Constant_1607 = makeConst(element::i64,
                                       ov::Shape({
                                           2,
                                       }),
                                       {0, 1});
        auto Unsqueeze_1608 = makeOP<opset1::Unsqueeze>({Unsqueeze_1606, Constant_1607});
        auto Convert_1795 = makeConst(
            element::f32,
            ov::Shape({
                1,
                1,
                1,
                1,
                1,
                3,
                3,
            }),
            {1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f});
        auto Multiply_1796 = makeOP<opset1::Multiply>({Unsqueeze_1608, Convert_1795}, {{"auto_broadcast", "numpy"}});
        auto Constant_1797 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {6});
        auto ReduceSum_1798 = makeOP<opset1::ReduceSum>({Multiply_1796, Constant_1797}, {{"keep_dims", false}});
        auto Constant_1801 = makeConst(element::i64,
                                       ov::Shape({
                                           6,
                                       }),
                                       {4, 0, 1, 2, 3, 5});
        auto Transpose_1802 = makeOP<opset1::Transpose>({ReduceSum_1798, Constant_1801});
        auto Multiply_1805 = makeOP<opset1::Multiply>({Unsqueeze_1804, Transpose_1802}, {{"auto_broadcast", "numpy"}});
        auto Constant_1994 = makeConst(element::i64,
                                       ov::Shape({
                                           6,
                                       }),
                                       {0, 5, 1, 2, 3, 4});
        auto Transpose_1995 = makeOP<opset1::Transpose>({Multiply_1805, Constant_1994});
        auto Concat_2043 = makeConst(element::i64,
                                     ov::Shape({
                                         6,
                                     }),
                                     {4, 3, 2, 1, 1, 3});
        auto Broadcast_2044 = makeOP<opset3::Broadcast>({Transpose_1995, Concat_2043}, {{"mode", "bidirectional"}});
        auto Concat_2076 = makeConst(element::i64,
                                     ov::Shape({
                                         4,
                                     }),
                                     {4, 3, 2, 3});
        auto Reshape_2077 = makeOP<opset1::Reshape>({Broadcast_2044, Concat_2076}, {{"special_zero", false}});
        auto MatMul_2116 =
            makeOP<opset1::MatMul>({Reshape_2052, Reshape_2077}, {{"transpose_a", true}, {"transpose_b", true}});
        auto Concat_2117 = makeConst(element::i64,
                                     ov::Shape({
                                         5,
                                     }),
                                     {4, 3, 2, 1, 1});
        auto Reshape_2118 = makeOP<opset1::Reshape>({MatMul_2116, Concat_2117}, {{"special_zero", false}});
        auto Constant_2119 = makeConst(element::i64,
                                       ov::Shape({
                                           5,
                                       }),
                                       {1, 2, 3, 4, 0});
        auto node_6 = makeOP<opset1::Transpose>({Reshape_2118, Constant_2119});
        model_ref = std::make_shared<Model>(NodeVector{node_6}, ParameterVector{node_4, node_2, node_0});
    }
}

TEST_F(TransformationTestsF, Einsum_3in_broadcast_duplicated_ellipsis_repeated_dynamic) {
    PartialShape data_shape_1 = PartialShape::dynamic(5);
    PartialShape data_shape_2 = PartialShape::dynamic(6);
    PartialShape data_shape_3 = PartialShape::dynamic(4);
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto data_3 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_3);
        auto einsum =
            std::make_shared<opset7::Einsum>(OutputVector{data_1, data_2, data_3}, "a...b,bcccdd,...dbcc->c...b");
        model = std::make_shared<Model>(NodeVector{einsum}, ParameterVector{data_1, data_2, data_3});
        manager.register_pass<ov::pass::EinsumDecomposition>();
    }
    {
        using namespace ov::gen_pattern;
        auto node_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_3);
        auto node_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto node_4 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto Constant_904 = makeConst(element::i64, ov::Shape({}), {0});
        auto ShapeOf_901 = makeOP<opset3::ShapeOf>({node_2}, {{"output_type", "i64"}});
        auto Constant_902 = makeConst(element::i64,
                                      ov::Shape({
                                          3,
                                      }),
                                      {1, 2, 3});
        auto Gather_906 = makeOP<opset7::Gather>({ShapeOf_901, Constant_902, Constant_904}, {{"batch_dims", 0}});
        auto ReduceProd_1318 = makeOP<opset1::ReduceProd>({Gather_906, Constant_904}, {{"keep_dims", true}});
        auto Constant_1319 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_1320 =
            makeOP<opset1::Broadcast>({Constant_904, ReduceProd_1318, Constant_1319}, {{"mode", "numpy"}});
        auto Gather_907 = makeOP<opset7::Gather>({Gather_906, Constant_904, Constant_904}, {{"batch_dims", 0}});
        auto Constant_903 = makeConst(element::i64, ov::Shape({}), {3});
        auto Power_908 = makeOP<opset1::Power>({Gather_907, Constant_903}, {{"auto_broadcast", "numpy"}});
        auto Constant_905 = makeConst(element::i64, ov::Shape({}), {1});
        auto Subtract_909 = makeOP<opset1::Subtract>({Power_908, Constant_905}, {{"auto_broadcast", "numpy"}});
        auto Maximum_912 = makeOP<opset1::Maximum>({Subtract_909, Constant_905}, {{"auto_broadcast", "numpy"}});
        auto Subtract_910 = makeOP<opset1::Subtract>({Gather_907, Constant_905}, {{"auto_broadcast", "numpy"}});
        auto Maximum_911 = makeOP<opset1::Maximum>({Subtract_910, Constant_905}, {{"auto_broadcast", "numpy"}});
        auto Divide_913 =
            makeOP<opset1::Divide>({Maximum_912, Maximum_911}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Range_914 = makeOP<opset1::Range>({Constant_904, Power_908, Divide_913});
        auto Unsqueeze_1313 = makeOP<opset1::Unsqueeze>({Gather_907, Constant_904});
        auto Constant_1314 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_1315 =
            makeOP<opset1::Broadcast>({Constant_905, Unsqueeze_1313, Constant_1314}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_1349 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_1320, Range_914, Broadcast_1315, Constant_904});
        auto ShapeOf_1350 = makeOP<opset1::ShapeOf>({ShapeOf_901});
        auto Constant_1351 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_1352 =
            makeOP<opset1::Broadcast>({Constant_905, ShapeOf_1350, Constant_1351}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_1355 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_1352, Constant_902, Gather_906, Constant_904});
        auto Reshape_1356 = makeOP<opset1::Reshape>({ScatterElementsUpdate_1349, ScatterElementsUpdate_1355},
                                                    {{"special_zero", false}});
        auto Constant_1361 = makeConst(element::i64, ov::Shape({}), {0});
        auto ShapeOf_1358 = makeOP<opset3::ShapeOf>({node_2}, {{"output_type", "i64"}});
        auto Constant_1359 = makeConst(element::i64,
                                       ov::Shape({
                                           2,
                                       }),
                                       {4, 5});
        auto Gather_1363 = makeOP<opset7::Gather>({ShapeOf_1358, Constant_1359, Constant_1361}, {{"batch_dims", 0}});
        auto ReduceProd_1775 = makeOP<opset1::ReduceProd>({Gather_1363, Constant_1361}, {{"keep_dims", true}});
        auto Constant_1776 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_1777 =
            makeOP<opset1::Broadcast>({Constant_1361, ReduceProd_1775, Constant_1776}, {{"mode", "numpy"}});
        auto Gather_1364 = makeOP<opset7::Gather>({Gather_1363, Constant_1361, Constant_1361}, {{"batch_dims", 0}});
        auto Constant_1360 = makeConst(element::i64, ov::Shape({}), {2});
        auto Power_1365 = makeOP<opset1::Power>({Gather_1364, Constant_1360}, {{"auto_broadcast", "numpy"}});
        auto Constant_1362 = makeConst(element::i64, ov::Shape({}), {1});
        auto Subtract_1366 = makeOP<opset1::Subtract>({Power_1365, Constant_1362}, {{"auto_broadcast", "numpy"}});
        auto Maximum_1369 = makeOP<opset1::Maximum>({Subtract_1366, Constant_1362}, {{"auto_broadcast", "numpy"}});
        auto Subtract_1367 = makeOP<opset1::Subtract>({Gather_1364, Constant_1362}, {{"auto_broadcast", "numpy"}});
        auto Maximum_1368 = makeOP<opset1::Maximum>({Subtract_1367, Constant_1362}, {{"auto_broadcast", "numpy"}});
        auto Divide_1370 =
            makeOP<opset1::Divide>({Maximum_1369, Maximum_1368}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Range_1371 = makeOP<opset1::Range>({Constant_1361, Power_1365, Divide_1370});
        auto Unsqueeze_1770 = makeOP<opset1::Unsqueeze>({Gather_1364, Constant_1361});
        auto Constant_1771 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_1772 =
            makeOP<opset1::Broadcast>({Constant_1362, Unsqueeze_1770, Constant_1771}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_1806 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_1777, Range_1371, Broadcast_1772, Constant_1361});
        auto ShapeOf_1807 = makeOP<opset1::ShapeOf>({ShapeOf_1358});
        auto Constant_1808 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_1809 =
            makeOP<opset1::Broadcast>({Constant_1362, ShapeOf_1807, Constant_1808}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_1812 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_1809, Constant_1359, Gather_1363, Constant_1361});
        auto Reshape_1813 = makeOP<opset1::Reshape>({ScatterElementsUpdate_1806, ScatterElementsUpdate_1812},
                                                    {{"special_zero", false}});
        auto Multiply_1815 = makeOP<opset1::Multiply>({Reshape_1356, Reshape_1813}, {{"auto_broadcast", "numpy"}});
        auto ConvertLike_1816 = makeOP<opset1::ConvertLike>({Multiply_1815, node_2});
        auto Multiply_1816 = makeOP<opset1::Multiply>({node_2, ConvertLike_1816}, {{"auto_broadcast", "numpy"}});
        auto Constant_1817 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {2, 3, 5});
        auto ReduceSum_1818 = makeOP<opset1::ReduceSum>({Multiply_1816, Constant_1817}, {{"keep_dims", false}});
        auto Constant_1833 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto ShapeOf_1821 = makeOP<opset3::ShapeOf>({ReduceSum_1818}, {{"output_type", "i64"}});
        auto Constant_1823 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Constant_1824 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Constant_1826 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1827 =
            makeOP<opset1::StridedSlice>({ShapeOf_1821, Constant_1823, Constant_1824, Constant_1826},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Broadcast_1834 = makeOP<opset3::Broadcast>({Constant_1833, StridedSlice_1827}, {{"mode", "numpy"}});
        auto Constant_894 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {0});
        auto ReduceSum_895 = makeOP<opset1::ReduceSum>({node_4, Constant_894}, {{"keep_dims", false}});
        auto Constant_898 = makeConst(element::i64,
                                      ov::Shape({
                                          2,
                                      }),
                                      {4, 5});
        auto Unsqueeze_899 = makeOP<opset1::Unsqueeze>({ReduceSum_895, Constant_898});
        auto Constant_430 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {0});
        auto Unsqueeze_431 = makeOP<opset1::Unsqueeze>({node_0, Constant_430});
        auto Constant_432 = makeConst(element::i64,
                                      ov::Shape({
                                          2,
                                      }),
                                      {0, 1});
        auto Unsqueeze_433 = makeOP<opset1::Unsqueeze>({Unsqueeze_431, Constant_432});
        auto Constant_437 = makeConst(element::i64, ov::Shape({}), {0});
        auto ShapeOf_434 = makeOP<opset3::ShapeOf>({Unsqueeze_433}, {{"output_type", "i64"}});
        auto Constant_435 = makeConst(element::i64,
                                      ov::Shape({
                                          2,
                                      }),
                                      {5, 6});
        auto Gather_439 = makeOP<opset7::Gather>({ShapeOf_434, Constant_435, Constant_437}, {{"batch_dims", 0}});
        auto ReduceProd_851 = makeOP<opset1::ReduceProd>({Gather_439, Constant_437}, {{"keep_dims", true}});
        auto Constant_852 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_853 =
            makeOP<opset1::Broadcast>({Constant_437, ReduceProd_851, Constant_852}, {{"mode", "numpy"}});
        auto Gather_440 = makeOP<opset7::Gather>({Gather_439, Constant_437, Constant_437}, {{"batch_dims", 0}});
        auto Constant_436 = makeConst(element::i64, ov::Shape({}), {2});
        auto Power_441 = makeOP<opset1::Power>({Gather_440, Constant_436}, {{"auto_broadcast", "numpy"}});
        auto Constant_438 = makeConst(element::i64, ov::Shape({}), {1});
        auto Subtract_442 = makeOP<opset1::Subtract>({Power_441, Constant_438}, {{"auto_broadcast", "numpy"}});
        auto Maximum_445 = makeOP<opset1::Maximum>({Subtract_442, Constant_438}, {{"auto_broadcast", "numpy"}});
        auto Subtract_443 = makeOP<opset1::Subtract>({Gather_440, Constant_438}, {{"auto_broadcast", "numpy"}});
        auto Maximum_444 = makeOP<opset1::Maximum>({Subtract_443, Constant_438}, {{"auto_broadcast", "numpy"}});
        auto Divide_446 =
            makeOP<opset1::Divide>({Maximum_445, Maximum_444}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Range_447 = makeOP<opset1::Range>({Constant_437, Power_441, Divide_446});
        auto Unsqueeze_846 = makeOP<opset1::Unsqueeze>({Gather_440, Constant_437});
        auto Constant_847 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_848 =
            makeOP<opset1::Broadcast>({Constant_438, Unsqueeze_846, Constant_847}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_882 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_853, Range_447, Broadcast_848, Constant_437});
        auto ShapeOf_883 = makeOP<opset1::ShapeOf>({ShapeOf_434});
        auto Constant_884 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_885 = makeOP<opset1::Broadcast>({Constant_438, ShapeOf_883, Constant_884}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_888 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_885, Constant_435, Gather_439, Constant_437});
        auto Reshape_889 =
            makeOP<opset1::Reshape>({ScatterElementsUpdate_882, ScatterElementsUpdate_888}, {{"special_zero", false}});
        auto ConvertLike_890 = makeOP<opset1::ConvertLike>({Reshape_889, Unsqueeze_433});
        auto Multiply_891 = makeOP<opset1::Multiply>({Unsqueeze_433, ConvertLike_890}, {{"auto_broadcast", "numpy"}});
        auto Constant_892 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {6});
        auto ReduceSum_893 = makeOP<opset1::ReduceSum>({Multiply_891, Constant_892}, {{"keep_dims", false}});
        auto Constant_896 = makeConst(element::i64,
                                      ov::Shape({
                                          6,
                                      }),
                                      {0, 1, 2, 4, 3, 5});
        auto Transpose_897 = makeOP<opset1::Transpose>({ReduceSum_893, Constant_896});
        auto Multiply_900 = makeOP<opset1::Multiply>({Unsqueeze_899, Transpose_897}, {{"auto_broadcast", "numpy"}});
        auto Constant_1819 = makeConst(element::i64,
                                       ov::Shape({
                                           6,
                                       }),
                                       {3, 5, 0, 1, 2, 4});
        auto Transpose_1820 = makeOP<opset1::Transpose>({Multiply_900, Constant_1819});
        auto ShapeOf_1822 = makeOP<opset3::ShapeOf>({Transpose_1820}, {{"output_type", "i64"}});
        auto Constant_1828 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Constant_1829 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Constant_1831 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1832 =
            makeOP<opset1::StridedSlice>({ShapeOf_1822, Constant_1828, Constant_1829, Constant_1831},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Broadcast_1837 =
            makeOP<opset3::Broadcast>({Broadcast_1834, StridedSlice_1832}, {{"mode", "bidirectional"}});
        auto ShapeOf_1840 = makeOP<opset3::ShapeOf>({Broadcast_1837}, {{"output_type", "i64"}});
        auto Constant_1851 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto Constant_1841 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Constant_1842 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {3});
        auto Constant_1844 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1845 =
            makeOP<opset1::StridedSlice>({ShapeOf_1821, Constant_1841, Constant_1842, Constant_1844},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Broadcast_1852 = makeOP<opset3::Broadcast>({Constant_1851, StridedSlice_1845}, {{"mode", "numpy"}});
        auto Constant_1846 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {5});
        auto Constant_1847 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {6});
        auto Constant_1849 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1850 =
            makeOP<opset1::StridedSlice>({ShapeOf_1822, Constant_1846, Constant_1847, Constant_1849},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Broadcast_1855 =
            makeOP<opset3::Broadcast>({Broadcast_1852, StridedSlice_1850}, {{"mode", "bidirectional"}});
        auto ShapeOf_1858 = makeOP<opset3::ShapeOf>({Broadcast_1855}, {{"output_type", "i64"}});
        auto Concat_1859 = makeOP<opset1::Concat>({ShapeOf_1840, ShapeOf_1858}, {{"axis", 0}});
        auto Broadcast_1860 = makeOP<opset3::Broadcast>({ReduceSum_1818, Concat_1859}, {{"mode", "bidirectional"}});
        auto ReduceProd_1875 = makeOP<opset1::ReduceProd>({ShapeOf_1858, {0}}, {{"keep_dims", true}});
        auto Constant_1873 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto Concat_1876 = makeOP<opset1::Concat>({ShapeOf_1840, ReduceProd_1875, Constant_1873}, {{"axis", 0}});
        auto Reshape_1903 = makeOP<opset1::Reshape>({Broadcast_1860, Concat_1876}, {{"special_zero", false}});
        auto Constant_1863 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Constant_1864 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {5});
        auto Constant_1866 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1867 =
            makeOP<opset1::StridedSlice>({ShapeOf_1822, Constant_1863, Constant_1864, Constant_1866},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Concat_1868 = makeOP<opset1::Concat>({ShapeOf_1840, StridedSlice_1867, ShapeOf_1858}, {{"axis", 0}});
        auto Broadcast_1869 = makeOP<opset3::Broadcast>({Transpose_1820, Concat_1868}, {{"mode", "bidirectional"}});
        auto Constant_1904 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto ReduceProd_1905 = makeOP<opset1::ReduceProd>({StridedSlice_1867, Constant_1904}, {{"keep_dims", true}});
        auto ReduceProd_1907 = makeOP<opset1::ReduceProd>({ShapeOf_1858, {0}}, {{"keep_dims", true}});
        auto Concat_1908 = makeOP<opset1::Concat>({ShapeOf_1840, ReduceProd_1905, ReduceProd_1907}, {{"axis", 0}});
        auto Reshape_1961 = makeOP<opset1::Reshape>({Broadcast_1869, Concat_1908}, {{"special_zero", false}});
        auto MatMul_1962 =
            makeOP<opset1::MatMul>({Reshape_1903, Reshape_1961}, {{"transpose_a", true}, {"transpose_b", true}});
        auto Concat_1963 = makeOP<opset1::Concat>({ShapeOf_1840, StridedSlice_1867}, {{"axis", 0}});
        auto Reshape_1964 = makeOP<opset1::Reshape>({MatMul_1962, Concat_1963}, {{"special_zero", false}});
        auto Constant_1965 = makeConst(element::i64,
                                       ov::Shape({
                                           5,
                                       }),
                                       {1, 2, 3, 4, 0});
        auto node_6 = makeOP<opset1::Transpose>({Reshape_1964, Constant_1965});
        model_ref = std::make_shared<Model>(NodeVector{node_6}, ParameterVector{node_0, node_2, node_4});
    }
}
