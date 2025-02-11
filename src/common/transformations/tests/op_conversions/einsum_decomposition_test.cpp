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
        auto Constant_489 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto Constant_490 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {2});
        auto Constant_492 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto reduced1 = makeOP<opset1::StridedSlice>({ShapeOf_data_1, Constant_489, Constant_490, Constant_492},
                                                     {{"begin_mask", {0}},
                                                      {"end_mask", {0}},
                                                      {"new_axis_mask", {}},
                                                      {"shrink_axis_mask", {}},
                                                      {"ellipsis_mask", {}}});
        // Get reduced subshape for data_2.
        auto Constant_494 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {2});
        auto Constant_495 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {3});
        auto Constant_497 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto reduced_2 = makeOP<opset1::StridedSlice>({ShapeOf_data_2, Constant_494, Constant_495, Constant_497},
                                                      {{"begin_mask", {0}},
                                                       {"end_mask", {0}},
                                                       {"new_axis_mask", {}},
                                                       {"shrink_axis_mask", {}},
                                                       {"ellipsis_mask", {}}});

        // broadcast_merge_shapes(reduced1, reduced_2)
        auto Constant_499 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto Broadcast_500 = makeOP<opset3::Broadcast>({Constant_499, reduced1}, {{"mode", "numpy"}});
        auto Broadcast_503 = makeOP<opset3::Broadcast>({Broadcast_500, reduced_2}, {{"mode", "bidirectional"}});
        auto reduced_subshape_broadcast_merge_shapes =
            makeOP<opset3::ShapeOf>({Broadcast_503}, {{"output_type", "i64"}});

        // Extract separate subshape for data_1.
        auto Constant_507 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {0});
        auto Constant_508 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto Constant_510 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto separate1_subshape =
            makeOP<opset1::StridedSlice>({ShapeOf_data_1, Constant_507, Constant_508, Constant_510},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});

        // Extract separate subshape for data_2.
        auto Constant_516 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {0});
        auto Constant_517 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {2});
        auto Constant_519 = makeConst(element::i64,
                                      ov::Shape({
                                          1,
                                      }),
                                      {1});
        auto separate2_subshape =
            makeOP<opset1::StridedSlice>({ShapeOf_data_2, Constant_516, Constant_517, Constant_519},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});

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
        auto Reduced1_subshape_red =
            makeOP<opset1::ReduceProd>({reduced_subshape_broadcast_merge_shapes, {0}}, {{"keep_dims", true}});
        // Merge subshapes
        auto reshape_subshape1 = makeOP<opset1::Concat>({Separate1_subshape_red, Reduced1_subshape_red}, {{"axis", 0}});
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
        auto node_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto node_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_2);
        auto Constant_1200 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Unsqueeze_1201 = makeOP<opset1::Unsqueeze>({node_2, Constant_1200});
        auto Constant_1202 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Unsqueeze_1203 = makeOP<opset1::Unsqueeze>({Unsqueeze_1201, Constant_1202});
        auto ShapeOf_1206 = makeOP<opset3::ShapeOf>({Unsqueeze_1203}, {{"output_type", "i64"}});
        auto Constant_1226 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Constant_1227 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto Constant_1229 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1230 =
            makeOP<opset1::StridedSlice>({ShapeOf_1206, Constant_1226, Constant_1227, Constant_1229},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Constant_1218 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto Constant_1208 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto Constant_1209 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {4});
        auto Constant_1211 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1212 =
            makeOP<opset1::StridedSlice>({ShapeOf_1206, Constant_1208, Constant_1209, Constant_1211},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Broadcast_1219 = makeOP<opset3::Broadcast>({Constant_1218, StridedSlice_1212}, {{"mode", "numpy"}});
        auto Constant_1204 = makeConst(element::i64,
                                       ov::Shape({
                                           5,
                                       }),
                                       {0, 4, 3, 1, 2});
        auto Transpose_1205 = makeOP<opset1::Transpose>({node_0, Constant_1204});
        auto ShapeOf_1207 = makeOP<opset3::ShapeOf>({Transpose_1205}, {{"output_type", "i64"}});
        auto Constant_1213 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Constant_1214 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {5});
        auto Constant_1216 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1217 =
            makeOP<opset1::StridedSlice>({ShapeOf_1207, Constant_1213, Constant_1214, Constant_1216},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Broadcast_1222 =
            makeOP<opset3::Broadcast>({Broadcast_1219, StridedSlice_1217}, {{"mode", "bidirectional"}});
        auto ShapeOf_1225 = makeOP<opset3::ShapeOf>({Broadcast_1222}, {{"output_type", "i64"}});
        auto Concat_1231 = makeOP<opset1::Concat>({StridedSlice_1230, ShapeOf_1225}, {{"axis", 0}});
        auto Broadcast_1232 = makeOP<opset3::Broadcast>({Unsqueeze_1203, Concat_1231}, {{"mode", "bidirectional"}});
        auto Constant_1244 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto ReduceProd_1245 = makeOP<opset1::ReduceProd>({StridedSlice_1230, Constant_1244}, {{"keep_dims", true}});
        auto ReduceProd_1247 = makeOP<opset1::ReduceProd>({ShapeOf_1225, {0}}, {{"keep_dims", true}});
        auto Concat_1248 = makeOP<opset1::Concat>({ReduceProd_1245, ReduceProd_1247}, {{"axis", 0}});
        auto Reshape_1249 = makeOP<opset1::Reshape>({Broadcast_1232, Concat_1248}, {{"special_zero", false}});
        auto Constant_1235 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto Constant_1236 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {2});
        auto Constant_1238 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {1});
        auto StridedSlice_1239 =
            makeOP<opset1::StridedSlice>({ShapeOf_1207, Constant_1235, Constant_1236, Constant_1238},
                                         {{"begin_mask", {0}},
                                          {"end_mask", {0}},
                                          {"new_axis_mask", {}},
                                          {"shrink_axis_mask", {}},
                                          {"ellipsis_mask", {}}});
        auto Concat_1240 = makeOP<opset1::Concat>({StridedSlice_1239, ShapeOf_1225}, {{"axis", 0}});
        auto Broadcast_1241 = makeOP<opset3::Broadcast>({Transpose_1205, Concat_1240}, {{"mode", "bidirectional"}});
        auto Constant_1302 = makeConst(element::i64,
                                       ov::Shape({
                                           1,
                                       }),
                                       {0});
        auto ReduceProd_1303 = makeOP<opset1::ReduceProd>({StridedSlice_1239, Constant_1302}, {{"keep_dims", true}});
        auto ReduceProd_1305 = makeOP<opset1::ReduceProd>({ShapeOf_1225, {0}}, {{"keep_dims", true}});
        auto Concat_1306 = makeOP<opset1::Concat>({ReduceProd_1303, ReduceProd_1305}, {{"axis", 0}});
        auto Reshape_1307 = makeOP<opset1::Reshape>({Broadcast_1241, Concat_1306}, {{"special_zero", false}});
        auto MatMul_1360 =
            makeOP<opset1::MatMul>({Reshape_1249, Reshape_1307}, {{"transpose_a", false}, {"transpose_b", true}});
        auto Concat_1361 = makeOP<opset1::Concat>({StridedSlice_1230, StridedSlice_1239}, {{"axis", 0}});
        auto Reshape_1362 = makeOP<opset1::Reshape>({MatMul_1360, Concat_1361}, {{"special_zero", false}});
        auto Constant_1363 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {1, 0, 2});
        auto node_4 = makeOP<opset1::Transpose>({Reshape_1362, Constant_1363});
        model_ref = std::make_shared<Model>(NodeVector{node_4}, ParameterVector{node_2, node_0});
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
        auto node_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto Multiply_1382 = makeConst(
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
        auto Multiply_1383 = makeOP<opset1::Multiply>({node_0, Multiply_1382}, {{"auto_broadcast", "numpy"}});
        auto Constant_1384 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {3, 4, 5});
        auto ReduceSum_1385 = makeOP<opset1::ReduceSum>({Multiply_1383, Constant_1384}, {{"keep_dims", false}});
        auto Constant_1386 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {1, 2, 0});
        auto node_2 = makeOP<opset1::Transpose>({ReduceSum_1385, Constant_1386});
        model_ref = std::make_shared<Model>(NodeVector{node_2}, ParameterVector{node_0});
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
        auto node_0 = std::make_shared<ov::op::v0::Parameter>(element::f32, data_shape_1);
        auto Constant_2112 = makeConst(element::i64, ov::Shape({}), {0});
        auto ShapeOf_2109 = makeOP<opset3::ShapeOf>({node_0}, {{"output_type", "i64"}});
        auto Constant_2110 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {0, 2, 4});
        auto Gather_2114 = makeOP<opset7::Gather>({ShapeOf_2109, Constant_2110, Constant_2112}, {{"batch_dims", 0}});
        auto ReduceProd_2526 = makeOP<opset1::ReduceProd>({Gather_2114, Constant_2112}, {{"keep_dims", true}});
        auto Constant_2527 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_2528 =
            makeOP<opset1::Broadcast>({Constant_2112, ReduceProd_2526, Constant_2527}, {{"mode", "numpy"}});
        auto Gather_2115 = makeOP<opset7::Gather>({Gather_2114, Constant_2112, Constant_2112}, {{"batch_dims", 0}});
        auto Constant_2111 = makeConst(element::i64, ov::Shape({}), {3});
        auto Power_2116 = makeOP<opset1::Power>({Gather_2115, Constant_2111}, {{"auto_broadcast", "numpy"}});
        auto Constant_2113 = makeConst(element::i64, ov::Shape({}), {1});
        auto Subtract_2117 = makeOP<opset1::Subtract>({Power_2116, Constant_2113}, {{"auto_broadcast", "numpy"}});
        auto Maximum_2120 = makeOP<opset1::Maximum>({Subtract_2117, Constant_2113}, {{"auto_broadcast", "numpy"}});
        auto Subtract_2118 = makeOP<opset1::Subtract>({Gather_2115, Constant_2113}, {{"auto_broadcast", "numpy"}});
        auto Maximum_2119 = makeOP<opset1::Maximum>({Subtract_2118, Constant_2113}, {{"auto_broadcast", "numpy"}});
        auto Divide_2121 =
            makeOP<opset1::Divide>({Maximum_2120, Maximum_2119}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Range_2122 = makeOP<opset1::Range>({Constant_2112, Power_2116, Divide_2121});
        auto Unsqueeze_2521 = makeOP<opset1::Unsqueeze>({Gather_2115, Constant_2112});
        auto Constant_2522 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_2523 =
            makeOP<opset1::Broadcast>({Constant_2113, Unsqueeze_2521, Constant_2522}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_2557 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_2528, Range_2122, Broadcast_2523, Constant_2112});
        auto ShapeOf_2558 = makeOP<opset1::ShapeOf>({ShapeOf_2109});
        auto Constant_2559 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_2560 =
            makeOP<opset1::Broadcast>({Constant_2113, ShapeOf_2558, Constant_2559}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_2563 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_2560, Constant_2110, Gather_2114, Constant_2112});
        auto Reshape_2564 = makeOP<opset1::Reshape>({ScatterElementsUpdate_2557, ScatterElementsUpdate_2563},
                                                    {{"special_zero", false}});
        auto Constant_2569 = makeConst(element::i64, ov::Shape({}), {0});
        auto ShapeOf_2566 = makeOP<opset3::ShapeOf>({node_0}, {{"output_type", "i64"}});
        auto Constant_2567 = makeConst(element::i64,
                                       ov::Shape({
                                           2,
                                       }),
                                       {1, 3});
        auto Gather_2571 = makeOP<opset7::Gather>({ShapeOf_2566, Constant_2567, Constant_2569}, {{"batch_dims", 0}});
        auto ReduceProd_2983 = makeOP<opset1::ReduceProd>({Gather_2571, Constant_2569}, {{"keep_dims", true}});
        auto Constant_2984 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_2985 =
            makeOP<opset1::Broadcast>({Constant_2569, ReduceProd_2983, Constant_2984}, {{"mode", "numpy"}});
        auto Gather_2572 = makeOP<opset7::Gather>({Gather_2571, Constant_2569, Constant_2569}, {{"batch_dims", 0}});
        auto Constant_2568 = makeConst(element::i64, ov::Shape({}), {2});
        auto Power_2573 = makeOP<opset1::Power>({Gather_2572, Constant_2568}, {{"auto_broadcast", "numpy"}});
        auto Constant_2570 = makeConst(element::i64, ov::Shape({}), {1});
        auto Subtract_2574 = makeOP<opset1::Subtract>({Power_2573, Constant_2570}, {{"auto_broadcast", "numpy"}});
        auto Maximum_2577 = makeOP<opset1::Maximum>({Subtract_2574, Constant_2570}, {{"auto_broadcast", "numpy"}});
        auto Subtract_2575 = makeOP<opset1::Subtract>({Gather_2572, Constant_2570}, {{"auto_broadcast", "numpy"}});
        auto Maximum_2576 = makeOP<opset1::Maximum>({Subtract_2575, Constant_2570}, {{"auto_broadcast", "numpy"}});
        auto Divide_2578 =
            makeOP<opset1::Divide>({Maximum_2577, Maximum_2576}, {{"auto_broadcast", "numpy"}, {"m_pythondiv", true}});
        auto Range_2579 = makeOP<opset1::Range>({Constant_2569, Power_2573, Divide_2578});
        auto Unsqueeze_2978 = makeOP<opset1::Unsqueeze>({Gather_2572, Constant_2569});
        auto Constant_2979 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_2980 =
            makeOP<opset1::Broadcast>({Constant_2570, Unsqueeze_2978, Constant_2979}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_3014 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_2985, Range_2579, Broadcast_2980, Constant_2569});
        auto ShapeOf_3015 = makeOP<opset1::ShapeOf>({ShapeOf_2566});
        auto Constant_3016 = makeConst(element::u8, ov::Shape({}), {0});
        auto Broadcast_3017 =
            makeOP<opset1::Broadcast>({Constant_2570, ShapeOf_3015, Constant_3016}, {{"mode", "numpy"}});
        auto ScatterElementsUpdate_3020 =
            makeOP<opset3::ScatterElementsUpdate>({Broadcast_3017, Constant_2567, Gather_2571, Constant_2569});
        auto Reshape_3021 = makeOP<opset1::Reshape>({ScatterElementsUpdate_3014, ScatterElementsUpdate_3020},
                                                    {{"special_zero", false}});
        auto Multiply_3023 = makeOP<opset1::Multiply>({Reshape_2564, Reshape_3021}, {{"auto_broadcast", "numpy"}});
        auto ConvertLike_3024 = makeOP<opset1::ConvertLike>({Multiply_3023, node_0});
        auto Multiply_3024 = makeOP<opset1::Multiply>({node_0, ConvertLike_3024}, {{"auto_broadcast", "numpy"}});
        auto Constant_3025 = makeConst(element::i64,
                                       ov::Shape({
                                           3,
                                       }),
                                       {2, 3, 4});
        auto ReduceSum_3026 = makeOP<opset1::ReduceSum>({Multiply_3024, Constant_3025}, {{"keep_dims", false}});
        auto Constant_3027 = makeConst(element::i64,
                                       ov::Shape({
                                           2,
                                       }),
                                       {1, 0});
        auto node_2 = makeOP<opset1::Transpose>({ReduceSum_3026, Constant_3027});
        model_ref = std::make_shared<Model>(NodeVector{node_2}, ParameterVector{node_0});
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
