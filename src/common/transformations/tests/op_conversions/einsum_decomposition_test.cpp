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
///
/// \brief Extracts the diagonal elements from the input tensor based on the specified repeated and unrepeated indices.
///
/// This function performs a series of operations to extract the diagonal elements from the input tensor `data`.
/// It first transposes the input tensor based on the repeated and unrepeated indices, then reshapes and pads the tensor
/// to isolate the diagonal elements. Finally, it gathers and squeezes the tensor to obtain the diagonal elements.
///
/// \param data A shared pointer to the input tensor node.
/// \param indices_of_repeated_labels A vector of vectors containing the indices of repeated labels.
/// \param unrepeated_indices A vector containing the indices of unrepeated labels. Default is an empty vector.
/// \return A shared pointer to the node representing the diagonal elements of the input tensor.
std::shared_ptr<ov::Node> extract_diagonal(const std::shared_ptr<ov::Node>& data,
                                           const std::vector<std::vector<size_t>>& indices_of_repeated_labels,
                                           const std::vector<size_t>& unrepeated_indices = {}) {
    std::vector<size_t> transpose_group_labels_target;
    std::vector<size_t> reduced_axes;

    // Prepare the target order for transposing the input tensor
    for (size_t i = 0; i < indices_of_repeated_labels.size(); i++) {
        auto repeated_label = indices_of_repeated_labels[i];
        size_t step = i * 2;
        reduced_axes.push_back(step + 1);
        transpose_group_labels_target.insert(transpose_group_labels_target.end(),
                                             repeated_label.begin(),
                                             repeated_label.end());
    }
    transpose_group_labels_target.insert(transpose_group_labels_target.end(),
                                         unrepeated_indices.begin(),
                                         unrepeated_indices.end());

    // Transpose the input tensor to group repeated and unrepeated labels
    auto const_transpose_group_labels_target =
        makeConst(element::i64, ov::Shape({transpose_group_labels_target.size()}), transpose_group_labels_target);
    auto transpose_group_labels = std::make_shared<ov::op::v1::Transpose>(data, const_transpose_group_labels_target);

    // Get the shape of the transposed tensor
    auto shapeof_transposed_data = std::make_shared<ov::op::v3::ShapeOf>(transpose_group_labels);

    auto const_0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {0});
    auto const_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {1});

    ov::NodeVector flattened_shapes;
    ov::NodeVector unflattened_shapes;
    ov::NodeVector begins;
    ov::NodeVector ends;

    std::vector<size_t> dim_iota(transpose_group_labels_target.size());
    std::iota(dim_iota.begin(), dim_iota.end(), 0);

    size_t dimension_iter = 0;

    // Process each repeated label group
    for (auto repeated_label : indices_of_repeated_labels) {
        auto num_repeats = repeated_label.size();
        std::vector<size_t> label_indices = {dim_iota.begin() + dimension_iter,
                                             dim_iota.begin() + dimension_iter + num_repeats};
        auto repeated_label_indices_len = ov::op::v0::Constant::create(ov::element::i64, {}, {num_repeats});
        auto repeated_label_indices =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape({num_repeats}), label_indices);
        auto repeated_dimensions =
            std::make_shared<ov::op::v7::Gather>(shapeof_transposed_data, repeated_label_indices, const_0);
        auto repeated_dimension = std::make_shared<ov::op::v7::Gather>(repeated_dimensions, const_0, const_0);
        auto range_max_val = std::make_shared<ov::op::v1::Power>(repeated_dimension, repeated_label_indices_len);
        auto step_numerator = std::make_shared<ov::op::v1::Subtract>(range_max_val, const_1);
        auto step_denominator = std::make_shared<ov::op::v1::Subtract>(repeated_dimension, const_1);
        auto step_denominator_but_not_0 = std::make_shared<ov::op::v1::Maximum>(step_denominator, const_1);
        auto step_numerator_but_not_0 = std::make_shared<ov::op::v1::Maximum>(step_numerator, const_1);
        auto step = std::make_shared<ov::op::v1::Divide>(step_numerator_but_not_0, step_denominator_but_not_0);
        auto end = std::make_shared<ov::op::v1::Subtract>(step, const_1);
        // Flatten all dimensions of single repeated label.
        auto reduced_size = std::make_shared<ov::op::v1::ReduceProd>(repeated_dimensions, const_0, true);
        flattened_shapes.push_back(reduced_size);
        // Reshape the tensor to restore the original shape with diagonal elements isolated and remainder.
        unflattened_shapes.push_back(repeated_dimension);
        unflattened_shapes.push_back(step);
        begins.push_back(const_0);
        ends.push_back(end);
        dimension_iter += num_repeats;
    }

    // Process unrepeated labels, do not perform flatten or pads on dimensions.
    std::vector<size_t> unrepeated_indices_after_transpose = {dim_iota.begin() + dimension_iter, dim_iota.end()};
    const auto& unrepeated_dimensions_indices =
        ov::op::v0::Constant::create(ov::element::i64,
                                     {unrepeated_indices_after_transpose.size()},
                                     unrepeated_indices_after_transpose);
    const auto unrepeated_dimensions =
        std::make_shared<ov::op::v7::Gather>(shapeof_transposed_data, unrepeated_dimensions_indices, const_0);
    begins.insert(begins.end(), unrepeated_indices_after_transpose.size(), const_0);
    ends.insert(ends.end(), unrepeated_indices_after_transpose.size(), const_0);
    flattened_shapes.push_back(unrepeated_dimensions);
    unflattened_shapes.push_back(unrepeated_dimensions);

    // Flatten the tensor to isolate diagonal elements
    auto flatten_labels_shape_target = std::make_shared<ov::op::v0::Concat>(flattened_shapes, 0);
    auto flatten_labels =
        std::make_shared<ov::op::v1::Reshape>(transpose_group_labels, flatten_labels_shape_target, false);

    // Pad the tensor to prepare for gathering diagonal elements
    auto pad_begin = std::make_shared<ov::op::v0::Concat>(begins, 0);
    auto pad_end = std::make_shared<ov::op::v0::Concat>(ends, 0);
    auto pad = std::make_shared<ov::op::v1::Pad>(flatten_labels, pad_begin, pad_end, ov::op::PadMode::CONSTANT);

    // Unflatten the tensor to restore the original shape with diagonal elements isolated
    auto unflatten_labels_shape_target = std::make_shared<ov::op::v0::Concat>(unflattened_shapes, 0);
    auto unflatten_labels = std::make_shared<ov::op::v1::Reshape>(pad, unflatten_labels_shape_target, false);

    // Gather the diagonal elements
    std::shared_ptr<ov::Node> gather = unflatten_labels;
    for (auto axis : reduced_axes) {
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape({1}), {axis});
        gather = std::make_shared<ov::op::v7::Gather>(gather, const_0, axis_const);
    }

    // Squeeze the tensor to remove the reduced dimensions
    auto squeeze_reduced_axes =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape({reduced_axes.size()}), reduced_axes);
    auto diagonal = std::make_shared<ov::op::v0::Squeeze>(gather, squeeze_reduced_axes);

    return diagonal;
}

}  // namespace

TEST_F(TransformationTestsF, Einsum_2in_matmul) {
    PartialShape data_shape_1{5, 2};
    PartialShape data_shape_2{10, 1, 25};
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
        auto const_0 = makeConst(element::i64, ov::Shape({1}), {0});
        auto const_1 = makeConst(element::i64, ov::Shape({1}), {1});
        auto const_3 = makeConst(element::i64, ov::Shape({1}), {3});
        // Transpose data so repeated labels are grouped and unrepeated labels are moved to back.
        // ij...iji -> iiijj...
        auto transpose_diagonal =
            makeOP<opset1::Transpose>({data_1, makeConst(element::i64, ov::Shape({6}), {0, 3, 5, 1, 4, 2})});
        // Flatten groups of repeated labels (ij...)
        auto flatten_repeated_labels =
            makeOP<opset1::Reshape>({transpose_diagonal, makeConst(element::i64, ov::Shape({3}), {1, 9, 2})},
                                    {{"special_zero", false}});
        // Pad begin and end are constant-folded.
        auto pad_begin = makeConst(element::i64, ov::Shape({3}), {0, 0, 0});
        auto pad_end = makeConst(element::i64, ov::Shape({3}), {0, 3, 0});
        auto pad = makeOP<opset1::Pad>(
            {flatten_repeated_labels, pad_begin, pad_end, makeConst(element::f32, ov::Shape({}), {0})},
            {{"pad_mode", "constant"}});
        // unflatten padded groups of repeated labels so i(padded reminder of i)j(padded reminder of j)...
        auto unflatten_repeated_labels =
            makeOP<opset1::Reshape>({pad, makeConst(element::i64, ov::Shape({5}), {1, 1, 3, 4, 2})},
                                    {{"special_zero", false}});
        // Reduce padded dimensions to get diagonal.
        auto reduce_first_repeat =
            makeOP<opset7::Gather>({unflatten_repeated_labels, const_0, const_1}, {{"batch_dims", 0}});
        auto reduce_second_repeat =
            makeOP<opset7::Gather>({reduce_first_repeat, const_0, const_3}, {{"batch_dims", 0}});
        auto remove_reduced_dims =
            makeOP<opset1::Squeeze>({reduce_second_repeat, makeConst(element::i64, ov::Shape({2}), {1, 3})});
        // Transpose to the original order of output labels.
        auto Constant_1386 = makeConst(element::i64, ov::Shape({3}), {1, 2, 0});
        auto transpose_out = makeOP<opset1::Transpose>({remove_reduced_dims, Constant_1386});
        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1});
    }
}

TEST_F(TransformationTestsF, Einsum_1in_repeated_labels_empty_ellipsis_dynamic) {
    PartialShape data_shape_1 = PartialShape::dynamic(5);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
        auto Constant_8230 = makeConst(element::i64, ov::Shape({6}), {1, 2, 3, 4, 5, 0});
        auto Transpose_8231 = makeOP<opset1::Transpose>({node_2, Constant_8230});
        auto Concat_8261 = makeConst(element::i64, ov::Shape({3}), {1, 1, 4});
        auto Reshape_8264 = makeOP<opset1::Reshape>({Transpose_8231, Concat_8261}, {{"special_zero", false}});
        auto Concat_8263 = makeConst(element::i64, ov::Shape({3}), {0, 0, 0});
        auto Concat_8262 = makeConst(element::i64, ov::Shape({3}), {0, 0, 0});
        auto Pad_8304 =
            makeOP<opset1::Pad>({Reshape_8264, Concat_8263, Concat_8262, 0.000000f}, {{"pad_mode", "constant"}});
        auto Concat_8381 = makeConst(element::i64, ov::Shape({5}), {1, 1, 1, 1, 4});
        auto Reshape_8382 = makeOP<opset1::Reshape>({Pad_8304, Concat_8381}, {{"special_zero", false}});
        auto Constant_8233 = makeConst(element::i64, ov::Shape({1}), {0});
        auto Constant_8459 = makeConst(element::i64, ov::Shape({1}), {1});
        auto Gather_8460 = makeOP<opset7::Gather>({Reshape_8382, Constant_8233, Constant_8459}, {{"batch_dims", 0}});
        auto Constant_8461 = makeConst(element::i64, ov::Shape({1}), {3});
        auto Gather_8462 = makeOP<opset7::Gather>({Gather_8460, Constant_8233, Constant_8461}, {{"batch_dims", 0}});
        auto Constant_8463 = makeConst(element::i64, ov::Shape({2}), {1, 3});
        auto Squeeze_8464 = makeOP<opset1::Squeeze>({Gather_8462, Constant_8463});
        auto Constant_8465 = makeConst(element::i64, ov::Shape({3}), {0, 2, 1});
        auto Transpose_8466 = makeOP<opset1::Transpose>({Squeeze_8464, Constant_8465});
        auto Constant_8494 = makeConst(element::i64, ov::Shape({3}), {3, 4, 3});
        auto Broadcast_8495 = makeOP<opset3::Broadcast>({Transpose_8466, Constant_8494}, {{"mode", "bidirectional"}});
        auto Constant_8528 = makeConst(element::i64, ov::Shape({4}), {3, 4, 1, 3});
        auto Reshape_8529 = makeOP<opset1::Reshape>({Broadcast_8495, Constant_8528}, {{"special_zero", false}});
        auto Constant_7971 = makeConst(element::i64, ov::Shape({6}), {0, 5, 1, 2, 3, 4});
        auto Transpose_7972 = makeOP<opset1::Transpose>({node_4, Constant_7971});
        auto Concat_7990 = makeConst(element::i64, ov::Shape({5}), {1, 2, 2, 1, 1});
        auto Reshape_7993 = makeOP<opset1::Reshape>({Transpose_7972, Concat_7990}, {{"special_zero", false}});
        auto Concat_7992 = makeConst(element::i64, ov::Shape({5}), {0, 0, 0, 0, 0});
        auto Concat_7991 = makeConst(element::i64, ov::Shape({5}), {0, 0, 0, 0, 0});
        auto Pad_8014 =
            makeOP<opset1::Pad>({Reshape_7993, Concat_7992, Concat_7991, 0.000000f}, {{"pad_mode", "constant"}});
        auto Concat_8053 = makeConst(element::i64, ov::Shape({6}), {1, 1, 2, 2, 1, 1});
        auto Reshape_8054 = makeOP<opset1::Reshape>({Pad_8014, Concat_8053}, {{"special_zero", false}});
        auto Constant_7974 = makeConst(element::i64, ov::Shape({1}), {0});
        auto Constant_8093 = makeConst(element::i64, ov::Shape({1}), {1});
        auto Gather_8094 = makeOP<opset7::Gather>({Reshape_8054, Constant_7974, Constant_8093}, {{"batch_dims", 0}});
        auto Constant_8095 = makeConst(element::i64, ov::Shape({1}), {1});
        auto Squeeze_8096 = makeOP<opset1::Squeeze>({Gather_8094, Constant_8095});
        auto Constant_8223 = makeConst(element::i64, ov::Shape({1}), {1});
        auto ReduceSum_8224 = makeOP<opset1::ReduceSum>({Squeeze_8096, Constant_8223}, {{"keep_dims", false}});
        auto Constant_8227 = makeConst(element::i64, ov::Shape({2}), {4, 5});
        auto Unsqueeze_8228 = makeOP<opset1::Unsqueeze>({ReduceSum_8224, Constant_8227});
        auto Constant_7967 = makeConst(element::i64, ov::Shape({1}), {0});
        auto Unsqueeze_7968 = makeOP<opset1::Unsqueeze>({node_0, Constant_7967});
        auto Constant_7969 = makeConst(element::i64, ov::Shape({2}), {0, 1});
        auto Unsqueeze_7970 = makeOP<opset1::Unsqueeze>({Unsqueeze_7968, Constant_7969});
        auto Constant_8097 = makeConst(element::i64, ov::Shape({7}), {5, 6, 0, 1, 2, 3, 4});
        auto Transpose_8098 = makeOP<opset1::Transpose>({Unsqueeze_7970, Constant_8097});
        auto Concat_8116 = makeConst(element::i64, ov::Shape({6}), {9, 1, 1, 1, 3, 1});
        auto Reshape_8119 = makeOP<opset1::Reshape>({Transpose_8098, Concat_8116}, {{"special_zero", false}});
        auto Concat_8118 = makeConst(element::i64, ov::Shape({6}), {0, 0, 0, 0, 0, 0});
        auto Concat_8117 = makeConst(element::i64, ov::Shape({6}), {3, 0, 0, 0, 0, 0});
        auto Pad_8140 =
            makeOP<opset1::Pad>({Reshape_8119, Concat_8118, Concat_8117, 0.000000f}, {{"pad_mode", "constant"}});
        auto Concat_8179 = makeConst(element::i64, ov::Shape({7}), {3, 4, 1, 1, 1, 3, 1});
        auto Reshape_8180 = makeOP<opset1::Reshape>({Pad_8140, Concat_8179}, {{"special_zero", false}});
        auto Constant_8100 = makeConst(element::i64, ov::Shape({1}), {0});
        auto Constant_8219 = makeConst(element::i64, ov::Shape({1}), {1});
        auto Gather_8220 = makeOP<opset7::Gather>({Reshape_8180, Constant_8100, Constant_8219}, {{"batch_dims", 0}});
        auto Constant_8221 = makeConst(element::i64, ov::Shape({1}), {1});
        auto Squeeze_8222 = makeOP<opset1::Squeeze>({Gather_8220, Constant_8221});
        auto Constant_8225 = makeConst(element::i64, ov::Shape({6}), {5, 1, 2, 3, 0, 4});
        auto Transpose_8226 = makeOP<opset1::Transpose>({Squeeze_8222, Constant_8225});
        auto Multiply_8229 = makeOP<opset1::Multiply>({Unsqueeze_8228, Transpose_8226}, {{"auto_broadcast", "numpy"}});
        auto Constant_8467 = makeConst(element::i64, ov::Shape({6}), {4, 0, 1, 2, 3, 5});
        auto Transpose_8468 = makeOP<opset1::Transpose>({Multiply_8229, Constant_8467});
        auto Constant_8502 = makeConst(element::i64, ov::Shape({6}), {3, 4, 2, 1, 1, 3});
        auto Broadcast_8503 = makeOP<opset3::Broadcast>({Transpose_8468, Constant_8502}, {{"mode", "bidirectional"}});
        auto Constant_8573 = makeConst(element::i64, ov::Shape({4}), {3, 4, 2, 3});
        auto Reshape_8574 = makeOP<opset1::Reshape>({Broadcast_8503, Constant_8573}, {{"special_zero", false}});
        auto MatMul_8575 =
            makeOP<opset1::MatMul>({Reshape_8529, Reshape_8574}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Constant_8577 = makeConst(element::i64, ov::Shape({5}), {3, 4, 2, 1, 1});
        auto Reshape_8578 = makeOP<opset1::Reshape>({MatMul_8575, Constant_8577}, {{"special_zero", false}});
        auto Constant_8579 = makeConst(element::i64, ov::Shape({5}), {0, 2, 3, 4, 1});
        auto node_6 = makeOP<opset1::Transpose>({Reshape_8578, Constant_8579});
        model_ref = std::make_shared<Model>(NodeVector{node_6}, ParameterVector{node_4, node_2, node_0});
    }
}

TEST_F(TransformationTestsF, Einsum_3in_broadcast_duplicated_ellipsis_repeated_dynamic) {
    PartialShape data_shape_1 = PartialShape::dynamic(5);
    PartialShape data_shape_2 = PartialShape::dynamic(6);
    PartialShape data_shape_3 = PartialShape::dynamic(4);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
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
        auto data_3_diagonal = extract_diagonal(data_3_processed, {{5, 6}}, {0, 1, 2, 3, 4});

        // No reduced labels - use simplified subgraph that uses Multiply instead Matmul
        // c...db -> ...bcd
        auto convenient_layout = makeConst(element::i64, ov::Shape({6}), {1, 2, 3, 5, 0, 4});
        auto rhs_convenient_layout = makeOP<opset1::Transpose>({data_3_diagonal, convenient_layout});
        // Optionally unsqueeze both operands for elementwise-multiplication with broadcasting
        // For LHS operand, unsqueeze at RHS separate dimensions indices (placed at end of RHS by transpose)
        auto lhs_unsqueeze_dims = makeConst(element::i64, ov::Shape({2}), {4, 5});
        auto lhs_unsqueeze = makeOP<opset1::Unsqueeze>({data_1_processed, lhs_unsqueeze_dims});
        // Out subscript = LHS_subscript + RHS_separate_part_subscript
        // ...bcd = ...b + cd
        auto data_1_3 = makeOP<opset1::Multiply>({lhs_unsqueeze, rhs_convenient_layout}, {{"auto_broadcast", "numpy"}});

        // Second pair of einsum inputs - data_2 and result of the first pair
        // bcccdd,...bcd->c...b
        // data_2 - handle repeated labels
        auto data_2_diagonal = extract_diagonal(data_2,
                                                {
                                                    {1, 2, 3},  // indices of repeated label c
                                                    {4, 5},     // indices_of_repeated_label_d
                                                },
                                                {
                                                    0,  // indices of unrepeated label b
                                                });
        // Transpose data_2 so that common labels, separated and reduced labels are grouped for both operands.
        auto data_2_processed =
            makeOP<opset1::Transpose>({data_2_diagonal, makeConst(element::i64, ov::Shape({3}), {0, 2, 1})});
        // data_1_3 - transpose to correctly group common, separate and reduced labels
        auto transpose_data_1_3_target = makeConst(element::i64, ov::Shape({6}), {4, 3, 0, 1, 2, 5});
        auto data_1_3_processed = makeOP<opset1::Transpose>({data_1_3, transpose_data_1_3_target});
        // Extract and broadcast common subshapes (bc)
        auto shapeof_data_1_3 = makeOP<opset3::ShapeOf>({data_1_3_processed}, {{"output_type", "i64"}});
        auto common_data_1_3 = extract_subshape_from_shape(shapeof_data_1_3, 0, 2);
        auto shapeof_data_2 = makeOP<opset3::ShapeOf>({data_2_processed}, {{"output_type", "i64"}});
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
            makeOP<opset3::Broadcast>({data_2_processed, broadcast_data_2_target}, {{"mode", "bidirectional"}});
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
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, separate_data_2_placeholder, reduced_prod},
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
            makeOP<opset1::MatMul>({reshape_data_2, reshape_data_1_3}, {{"transpose_a", false}, {"transpose_b", true}});
        auto reshape_out_subshape =
            makeOP<opset1::Concat>({common_broadcast_merge_shapes, separate_data_1_3}, {{"axis", 0}});
        auto reshape_out = makeOP<opset1::Reshape>({matmul, reshape_out_subshape}, {{"special_zero", false}});
        auto Constant_1965 = makeConst(element::i64, ov::Shape({5}), {0, 2, 3, 4, 1});
        auto transpose_out = makeOP<opset1::Transpose>({reshape_out, Constant_1965});
        model_ref = std::make_shared<Model>(NodeVector{transpose_out}, ParameterVector{data_1, data_2, data_3});
    }
}
