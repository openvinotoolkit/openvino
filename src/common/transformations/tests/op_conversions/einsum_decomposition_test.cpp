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
    auto broadcasted_shapes = makeOP<opset1::Maximum>({shape_node_lhs, shape_node_rhs}, {{"auto_broadcast", "numpy"}});
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
        auto Constant_485 = makeConst(element::i64, ov::Shape({3}), {0, 2, 1});
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
        auto Constant_525 = makeConst(element::i64, ov::Shape({1}), {0});
        // Reduce separate and reduced
        auto Separate1_subshape_red =
            makeOP<opset1::ReduceProd>({separate1_subshape, Constant_525}, {{"keep_dims", true}});
        auto reduced1_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        // Merge subshapes
        auto reshape_subshape1 = makeOP<opset1::Concat>({Separate1_subshape_red, reduced1_subshape_red}, {{"axis", 0}});
        auto Reshape_1 = makeOP<opset1::Reshape>({Broadcast_data_1, reshape_subshape1}, {{"special_zero", false}});
        // Reshape 2
        auto Constant_569 = makeConst(element::i64, ov::Shape({1}), {0});
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
        auto Constant_1200 = makeConst(element::i64, ov::Shape({1}), {2});
        auto Unsqueeze_1201 = makeOP<opset1::Unsqueeze>({data_1, Constant_1200});
        // Match ranks of dimensions covered by ellipsis labels
        auto Constant_1202 = makeConst(element::i64, ov::Shape({1}), {2});
        auto data_1_processed = makeOP<opset1::Unsqueeze>({Unsqueeze_1201, Constant_1202});
        // Process data_2
        // Transpose data_2 so that common labels, separated and reduced labels are grouped for both operands.
        auto Constant_1204 = makeConst(element::i64, ov::Shape({5}), {0, 4, 3, 1, 2});
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
        auto Constant_1244 = makeConst(element::i64, ov::Shape({1}), {0});
        auto Separate1_subshape_red =
            makeOP<opset1::ReduceProd>({separate1_subshape, Constant_1244}, {{"keep_dims", true}});
        auto reduced1_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        auto reshape1_shape = makeOP<opset1::Concat>({Separate1_subshape_red, reduced1_subshape_red}, {{"axis", 0}});
        auto Reshape_1 = makeOP<opset1::Reshape>({Broadcast_data_1, reshape1_shape}, {{"special_zero", false}});

        // Reshape 2
        auto Constant_1302 = makeConst(element::i64, ov::Shape({1}), {0});
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
        auto Constant_1363 = makeConst(element::i64, ov::Shape({3}), {1, 0, 2});
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
            ov::Shape({1, 3, 1, 1, 3, 1}),
            {1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f});
        auto Multiply_1383 = makeOP<opset1::Multiply>({data_1, multi_identity}, {{"auto_broadcast", "numpy"}});
        auto Constant_1384 = makeConst(element::i64, ov::Shape({3}), {3, 4, 5});
        auto data_1_diagonal = makeOP<opset1::ReduceSum>({Multiply_1383, Constant_1384}, {{"keep_dims", false}});
        // Transpose to the original order of output labels.
        auto Constant_1386 = makeConst(element::i64, ov::Shape({3}), {1, 2, 0});
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
        auto Constant_3027 = makeConst(element::i64, ov::Shape({2}), {1, 0});
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
        // ConstantFold folded multi-identity for input 2 to single constant
        auto Multiply_1990 = makeConst(element::f32, ov::Shape({1, 1, 1, 1, 1, 1}), {1.000000f});
        // Extract diagonals
        auto Multiply_1991 = makeOP<opset1::Multiply>({node_2, Multiply_1990}, {{"auto_broadcast", "numpy"}});
        auto Constant_1992 = makeConst(element::i64, ov::Shape({3}), {2, 3, 5});
        auto ReduceSum_1993 = makeOP<opset1::ReduceSum>({Multiply_1991, Constant_1992}, {{"keep_dims", false}});
        // Broadcast for ellipsis and labels constant folded to single constant and broadcast
        auto Concat_2034 = makeConst(element::i64, ov::Shape({3}), {4, 3, 3});
        // Broadcast ellipsis and labels
        auto Broadcast_2035 = makeOP<opset3::Broadcast>({ReduceSum_1993, Concat_2034}, {{"mode", "bidirectional"}});
        auto Concat_2051 = makeConst(element::i64, ov::Shape({4}), {4, 3, 3, 1});
        auto Reshape_2052 = makeOP<opset1::Reshape>({Broadcast_2035, Concat_2051}, {{"special_zero", false}});
        auto Convert_1700 = makeConst(element::f32, ov::Shape({1, 1, 1, 1, 1, 1}), {1.000000f});
        auto Multiply_1701 = makeOP<opset1::Multiply>({node_4, Convert_1700}, {{"auto_broadcast", "numpy"}});
        auto Constant_1702 = makeConst(element::i64, ov::Shape({1}), {5});
        auto ReduceSum_1703 = makeOP<opset1::ReduceSum>({Multiply_1701, Constant_1702}, {{"keep_dims", false}});
        auto Constant_1799 = makeConst(element::i64, ov::Shape({1}), {1});
        auto ReduceSum_1800 = makeOP<opset1::ReduceSum>({ReduceSum_1703, Constant_1799}, {{"keep_dims", false}});
        auto Constant_1803 = makeConst(element::i64, ov::Shape({2}), {4, 5});
        auto Unsqueeze_1804 = makeOP<opset1::Unsqueeze>({ReduceSum_1800, Constant_1803});
        auto Constant_1605 = makeConst(element::i64, ov::Shape({1}), {0});
        auto Unsqueeze_1606 = makeOP<opset1::Unsqueeze>({node_0, Constant_1605});
        auto Constant_1607 = makeConst(element::i64, ov::Shape({2}), {0, 1});
        auto Unsqueeze_1608 = makeOP<opset1::Unsqueeze>({Unsqueeze_1606, Constant_1607});
        auto Convert_1795 = makeConst(
            element::f32,
            ov::Shape({1, 1, 1, 1, 1, 3, 3}),
            {1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.000000f, 0.000000f, 1.000000f});
        auto Multiply_1796 = makeOP<opset1::Multiply>({Unsqueeze_1608, Convert_1795}, {{"auto_broadcast", "numpy"}});
        auto Constant_1797 = makeConst(element::i64, ov::Shape({1}), {6});
        auto ReduceSum_1798 = makeOP<opset1::ReduceSum>({Multiply_1796, Constant_1797}, {{"keep_dims", false}});
        auto Constant_1801 = makeConst(element::i64, ov::Shape({6}), {4, 0, 1, 2, 3, 5});
        auto Transpose_1802 = makeOP<opset1::Transpose>({ReduceSum_1798, Constant_1801});
        auto Multiply_1805 = makeOP<opset1::Multiply>({Unsqueeze_1804, Transpose_1802}, {{"auto_broadcast", "numpy"}});
        auto Constant_1994 = makeConst(element::i64, ov::Shape({6}), {0, 5, 1, 2, 3, 4});
        auto Transpose_1995 = makeOP<opset1::Transpose>({Multiply_1805, Constant_1994});
        auto Concat_2043 = makeConst(element::i64, ov::Shape({6}), {4, 3, 2, 1, 1, 3});
        auto Broadcast_2044 = makeOP<opset3::Broadcast>({Transpose_1995, Concat_2043}, {{"mode", "bidirectional"}});
        auto Concat_2076 = makeConst(element::i64, ov::Shape({4}), {4, 3, 2, 3});
        auto Reshape_2077 = makeOP<opset1::Reshape>({Broadcast_2044, Concat_2076}, {{"special_zero", false}});
        auto MatMul_2116 =
            makeOP<opset1::MatMul>({Reshape_2052, Reshape_2077}, {{"transpose_a", true}, {"transpose_b", true}});
        auto Concat_2117 = makeConst(element::i64, ov::Shape({5}), {4, 3, 2, 1, 1});
        auto Reshape_2118 = makeOP<opset1::Reshape>({MatMul_2116, Concat_2117}, {{"special_zero", false}});
        auto Constant_2119 = makeConst(element::i64, ov::Shape({5}), {1, 2, 3, 4, 0});
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
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto data_3 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_3);

        // First pair of einsum inputs - data_1 and data_3
        // data_1 - label `a` can be reduced by reduce_input()
        auto indice_of_a_in_data_1 = makeConst(element::i64, ov::Shape({1}), {0});
        auto data_1_processed = makeOP<opset1::ReduceSum>({data_1, indice_of_a_in_data_1}, {{"keep_dims", false}});
        // data_3 - unsqueeze ellipse labels to allow for broadcasting and handle repeated labels
        auto ellipsis_idx = makeConst(element::i64, ov::Shape({1}), {0});
        auto data3_insert_missing_ellipsis = makeOP<opset1::Unsqueeze>({data_3, ellipsis_idx});
        auto align_ellipsis_idx = makeConst(element::i64, ov::Shape({2}), {0, 1});
        auto data_3_processed = makeOP<opset1::Unsqueeze>({data3_insert_missing_ellipsis, align_ellipsis_idx});
        auto data_3_diagonal = extract_diagonal(data_3_processed, {{5, 6}});

        // No reduced labels - use simplified subgraph that uses Multiply instead Matmul
        auto convenient_layout = makeConst(element::i64, ov::Shape({6}), {0, 1, 2, 4, 3, 5});
        // ...dbc -> ...bdc
        auto rhs_convenient_layout = makeOP<opset1::Transpose>({data_3_diagonal, convenient_layout});
        // Optionally unsqueeze both operands for elementwise-multiplication with broadcasting
        // For LHS operand, unsqueeze at RHS separate dimensions indices (placed at end of RHS by transpose)
        auto lhs_unsqueeze_dims = makeConst(element::i64, ov::Shape({2}), {4, 5});
        auto lhs_unsqueeze = makeOP<opset1::Unsqueeze>({data_1_processed, lhs_unsqueeze_dims});
        // Out subscript = LHS_subscript + RHS_separate_part_subscript
        // ...bdc = ...b + dc
        auto data_1_3 = makeOP<opset1::Multiply>({lhs_unsqueeze, rhs_convenient_layout}, {{"auto_broadcast", "numpy"}});

        // Second pair of einsum inputs - data_2 and result of the first pair
        // bcccdd,...bdc->c...b
        // data_2 - handle repeated labels
        auto data_2_diagonal = extract_diagonal(data_2,
                                                {
                                                    {1, 2, 3},  // indices of repeated label c
                                                    {4, 5},     // indices_of_repeated_label_d
                                                });
        // data_1_3 - transpose to correctly group common, separate and reduced labels
        // ...bdc->bc...d
        auto transpose_data_1_3_target = makeConst(element::i64, ov::Shape({6}), {3, 5, 0, 1, 2, 4});
        auto data_1_3_processed = makeOP<opset1::Transpose>({data_1_3, transpose_data_1_3_target});
        // Extract and broadcast common subshapes (bc)
        auto shapeof_data_1_3 = makeOP<opset3::ShapeOf>({data_1_3_processed}, {{"output_type", "i64"}});
        auto common_data_1_3 = extract_subshape_from_shape(shapeof_data_1_3, 0, 2);
        auto shapeof_data_2 = makeOP<opset3::ShapeOf>({data_2_diagonal}, {{"output_type", "i64"}});
        auto common_data_2 = extract_subshape_from_shape(shapeof_data_2, 0, 2);
        auto common_broadcast_merge_shapes = broadcast_merge_shapes(common_data_2, common_data_1_3);

        // Extract and broadcast reduced subshapes (d)
        auto reduced_data_2 = extract_subshape_from_shape(shapeof_data_2, 2, 3);
        auto reduced_data_1_3 = extract_subshape_from_shape(shapeof_data_1_3, 5, 6);
        auto reduced_broadcast_merge_shapes = broadcast_merge_shapes(reduced_data_2, reduced_data_1_3);

        // Extract and broadcast separate subshapes if needed
        auto separate_data_1_3 = extract_subshape_from_shape(shapeof_data_1_3, 2, 5);  // (...)

        // Broadcast data_2 and data_1_3 based on calculated subshapes
        auto broadcast_data_2_target =
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, reduced_broadcast_merge_shapes}, {{"axis", 0}});
        auto broadcast_data_2 =
            makeOP<opset3::Broadcast>({data_2_diagonal, broadcast_data_2_target}, {{"mode", "bidirectional"}});
        auto broadcast_data_1_3_target =
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, separate_data_1_3, reduced_broadcast_merge_shapes},
                                   {{"axis", 0}});
        auto broadcast_data_1_3 =
            makeOP<opset3::Broadcast>({data_1_3_processed, broadcast_data_1_3_target}, {{"mode", "bidirectional"}});

        // Optionally reshape broadcasted data_2 and data_1_3 so separate and reduced labels are represented by one
        // dimension. Subgraphes are constant-folded, target subshapes are calculated broadcast_merge_shapes function.
        auto reduced_prod = makeOP<opset1::ReduceProd>({reduced_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        // Reshape data_2
        auto separate_data_2_placeholder = makeConst(element::i64, ov::Shape({1}), {1});
        auto reshape_data_2_target =
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, reduced_prod, separate_data_2_placeholder},
                                   {{"axis", 0}});
        auto reshape_data_2 =
            makeOP<opset1::Reshape>({broadcast_data_2, reshape_data_2_target}, {{"special_zero", false}});
        // Reshape data_1_3
        auto Constant_1904 = makeConst(element::i64, ov::Shape({1}), {0});
        auto separate_data_1_3_prod =
            makeOP<opset1::ReduceProd>({separate_data_1_3, Constant_1904}, {{"keep_dims", true}});
        auto reshape_data_1_3_target =
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, separate_data_1_3_prod, reduced_prod},
                                   {{"axis", 0}});
        auto reshape_data_1_3 =
            makeOP<opset1::Reshape>({broadcast_data_1_3, reshape_data_1_3_target}, {{"special_zero", false}});
        auto matmul =
            makeOP<opset1::MatMul>({reshape_data_2, reshape_data_1_3}, {{"transpose_a", true}, {"transpose_b", true}});
        auto reshape_out_subshape =
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, separate_data_1_3}, {{"axis", 0}});
        auto reshape_out = makeOP<opset1::Reshape>({matmul, reshape_out_subshape}, {{"special_zero", false}});
        auto Constant_1965 = makeConst(element::i64, ov::Shape({5}), {1, 2, 3, 4, 0});
        auto transpose_out = makeOP<opset1::Transpose>({reshape_out, Constant_1965});
        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1, data_2, data_3});
    }
}
