// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/propagate_slice.hpp"

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"

using namespace ov;
using namespace ov::op;

// Helper to count Slice nodes
static size_t count_slice_nodes(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ops()) {
        if (std::dynamic_pointer_cast<v8::Slice>(node)) {
            count++;
        }
    }
    return count;
}

// Helper to run PropagateSliceUp on a model
static void apply_propagate_slice_up(const std::shared_ptr<ov::Model>& model) {
    ov::npuw::PropagateSliceUp pass;
    pass.run_on_model(model);
}

// Helper to build a single-axis Slice with explicit start/stop/step (default step=1) —
// used by tests that slice an explicit sub-range along one axis.
static std::shared_ptr<v8::Slice> make_slice(const Output<Node>& data,
                                             int64_t axis,
                                             int64_t start,
                                             int64_t stop,
                                             int64_t step = 1) {
    auto start_c = v0::Constant::create(element::i64, Shape{1}, {start});
    auto stop_c = v0::Constant::create(element::i64, Shape{1}, {stop});
    auto step_c = v0::Constant::create(element::i64, Shape{1}, {step});
    auto axes_c = v0::Constant::create(element::i64, Shape{1}, {axis});
    return std::make_shared<v8::Slice>(data, start_c, stop_c, step_c, axes_c);
}

// Helper to build a single-axis Slice selecting the last index along `axis`
// (start=-1, stop=INT64_MAX, step=1) — the common "reduce this axis to size 1"
// pattern used across most propagation tests.
static std::shared_ptr<v8::Slice> make_last_index_slice(const Output<Node>& data, int64_t axis) {
    return make_slice(data, axis, -1, INT64_MAX, 1);
}

// Helper to build a multi-axis Slice with explicit per-axis start/stop (step defaults to 1 for all axes).
static std::shared_ptr<v8::Slice> make_multi_axis_slice(const Output<Node>& data,
                                                        const std::vector<int64_t>& axes,
                                                        const std::vector<int64_t>& starts,
                                                        const std::vector<int64_t>& stops) {
    std::vector<int64_t> steps(axes.size(), 1);
    auto start_c = v0::Constant::create(element::i64, Shape{axes.size()}, starts);
    auto stop_c = v0::Constant::create(element::i64, Shape{axes.size()}, stops);
    auto step_c = v0::Constant::create(element::i64, Shape{axes.size()}, steps);
    auto axes_c = v0::Constant::create(element::i64, Shape{axes.size()}, axes);
    return std::make_shared<v8::Slice>(data, start_c, stop_c, step_c, axes_c);
}

// Test R1: Slice(Gelu(X)) -> Gelu(Slice(X))
TEST(PropagateSliceTest, PropagateSliceThroughUnary_Gelu) {
    // Build: Param[1,1024,3072] -> Gelu -> Slice(axis=1, 1024->1) -> Result[1,1,3072]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto gelu = std::make_shared<v0::Gelu>(param);

    auto slice = make_last_index_slice(gelu, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Slice should have moved before Gelu
    // The result's input should be Gelu, and Gelu's input should be Slice
    auto result_node = model->get_results()[0];
    auto gelu_output = result_node->input_value(0);
    ASSERT_TRUE(is_type<v0::Gelu>(gelu_output.get_node_shared_ptr()));

    auto slice_output = gelu_output.get_node_shared_ptr()->input_value(0);
    EXPECT_TRUE(is_type<v8::Slice>(slice_output.get_node_shared_ptr()));

    // Check output shape
    EXPECT_EQ(result_node->get_input_shape(0), (Shape{1, 1, 3072}));
}

// Test R2a: Slice(Add(A, B)) -> Add(Slice(A), Slice(B))
TEST(PropagateSliceTest, PropagateSliceThroughBinary_Add_BothSliced) {
    // Build: Add([1,1024,3072], [1,1024,3072]) -> Slice(axis=1) -> Result[1,1,3072]
    auto param1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto param2 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto add = std::make_shared<v1::Add>(param1, param2);

    auto slice = make_last_index_slice(add, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param1, param2});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Should have 2 Slice nodes (one on each input of Add)
    EXPECT_EQ(count_slice_nodes(model), 2);

    // Result's input should be Add
    auto result_node = model->get_results()[0];
    EXPECT_TRUE(is_type<v1::Add>(result_node->input_value(0).get_node_shared_ptr()));
}

// Test R2b: Slice(Add(A, broadcast_B)) -> Add(Slice(A), broadcast_B)
TEST(PropagateSliceTest, PropagateSliceThroughBinary_Add_OneBroadcast) {
    // Build: Add([1,1024,3072], [1,1,3072]) -> Slice(axis=1) -> Result[1,1,3072]
    auto param1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto param2 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1, 3072});
    auto add = std::make_shared<v1::Add>(param1, param2);

    auto slice = make_last_index_slice(add, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param1, param2});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Should have only 1 Slice node (on non-broadcast input)
    EXPECT_EQ(count_slice_nodes(model), 1);
}

// Test R4: Slice(ReduceSum(X, axis=0)) with keep_dims=False
TEST(PropagateSliceTest, PropagateSliceThroughReduce_ReduceSum) {
    // Build: Param[8,1024,2048] -> ReduceSum(axis=0) -> [1024,2048] -> Slice(axis=0) -> [1,2048]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{8, 1024, 2048});
    auto axes = v0::Constant::create(element::i64, Shape{1}, {0});
    auto reduce = std::make_shared<v1::ReduceSum>(param, axes, false);  // keep_dims=false

    auto slice = make_last_index_slice(reduce, 0);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Slice should have moved before ReduceSum on input axis=1
    auto result_node = model->get_results()[0];
    auto reduce_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v1::ReduceSum>(reduce_node));

    auto slice_node = reduce_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));

    // Check output shape: [8,1,2048] -> ReduceSum(axis=0) -> [1,2048]
    EXPECT_EQ(result_node->get_input_shape(0), (Shape{1, 2048}));
}

// Test R5: Slice(MatMul(X, W))
TEST(PropagateSliceTest, PropagateSliceThroughMatMul) {
    // Build: Param[1,1024,3072] -> MatMul(W[3072,2048]) -> [1,1024,2048] -> Slice(axis=1) -> [1,1,2048]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto weight = v0::Constant::create(element::f32, Shape{3072, 2048}, {0.1f});
    auto matmul = std::make_shared<v0::MatMul>(param, weight);

    auto slice = make_last_index_slice(matmul, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Slice should have moved before MatMul
    auto result_node = model->get_results()[0];
    auto matmul_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v0::MatMul>(matmul_node));

    auto slice_node = matmul_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));
}

// Test R6: Slice(Reshape(X)) - squeeze-like with Unsqueeze
TEST(PropagateSliceTest, PropagateSliceThroughReshape_SqueezeLike) {
    // Build: Param[1024,2048] -> Reshape([1,1024,2048]) -> Slice(axis=1) -> [1,1,2048]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 2048});
    auto pattern = v0::Constant::create(element::i64, Shape{3}, {1, 1024, 2048});
    auto reshape = std::make_shared<v1::Reshape>(param, pattern, false);

    auto slice = make_last_index_slice(reshape, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Should have Unsqueeze instead of Reshape
    auto result_node = model->get_results()[0];
    auto unsqueeze_node = result_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Unsqueeze>(unsqueeze_node));

    // Unsqueeze input should be Slice
    auto slice_node = unsqueeze_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));
}

// Test R7: Slice(Transpose(X))
TEST(PropagateSliceTest, PropagateSliceThroughTranspose) {
    // Build: Param[1,32,1024,96] -> Transpose(0,2,1,3) -> [1,1024,32,96] -> Slice(axis=1) -> [1,1,32,96]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 32, 1024, 96});
    auto perm = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    auto transpose = std::make_shared<v1::Transpose>(param, perm);

    auto slice = make_last_index_slice(transpose, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Slice should have moved before Transpose (on input axis=2)
    auto result_node = model->get_results()[0];
    auto transpose_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v1::Transpose>(transpose_node));

    auto slice_node = transpose_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));
}

// Test R9: Merge duplicate Slices
TEST(PropagateSliceTest, MergeDuplicateSlices) {
    // Build: Param -> Gelu -> Split2 branches -> identical Slices on each branch
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto gelu = std::make_shared<v0::Gelu>(param);

    // Create two identical slices
    auto slice1 = make_last_index_slice(gelu, 1);
    auto slice2 = make_last_index_slice(gelu, 1);

    auto add = std::make_shared<v1::Add>(slice1, slice2);
    auto result = std::make_shared<v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Before: 2 Slice nodes
    ASSERT_EQ(count_slice_nodes(model), 2);

    // Apply pass
    apply_propagate_slice_up(model);

    // After: Should merge to 1 Slice node
    EXPECT_EQ(count_slice_nodes(model), 1);
}

// Test R9b: Do NOT merge slices from different output ports (e.g., TopK values vs indices)
TEST(PropagateSliceTest, DoNotMergeSlicesFromDifferentOutputPorts) {
    // Build: Param -> TopK(axis=1) -> [values(f32), indices(i64)] each with an identical Slice
    // on axis=1 (the TopK axis itself). Slicing the TopK axis would change the semantics of TopK
    // (it must see the full k-sized dimension), so PropagateSliceThroughTopK explicitly refuses
    // to hoist this Slice above TopK -- unlike slicing any other axis, which IS a legal, intended
    // optimization performed by that rule (and would legitimately collapse both Slices into a
    // single one placed before TopK, which is not what this test is about).
    // With propagation blocked, this test verifies the separate invariant that MergeDuplicateSlices
    // never merges two Slices that consume different output ports of the same node, even when
    // their slice parameters are identical.
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 128});
    auto k = v0::Constant::create(element::i64, Shape{}, {8});
    auto topk = std::make_shared<v3::TopK>(param, k, 1, v3::TopK::Mode::MAX, v3::TopK::SortType::SORT_VALUES);

    // Identical slice parameters on both outputs, slicing the TopK axis (axis=1): k=8 -> 4
    auto slice_values = make_slice(topk->output(0), 1, 0, 4);   // from values (f32)
    auto slice_indices = make_slice(topk->output(1), 1, 0, 4);  // from indices (i64)

    auto result1 = std::make_shared<v0::Result>(slice_values);
    auto result2 = std::make_shared<v0::Result>(slice_indices);
    auto model = std::make_shared<ov::Model>(ResultVector{result1, result2}, ParameterVector{param});

    // Before: 2 Slice nodes
    ASSERT_EQ(count_slice_nodes(model), 2);

    // Apply pass
    apply_propagate_slice_up(model);

    // After: Should still have 2 Slice nodes, each still consuming its own TopK output
    // (no propagation above TopK since the sliced axis is the TopK axis itself,
    // and no merge across the two different output ports)
    EXPECT_EQ(count_slice_nodes(model), 2);

    auto result1_input = model->get_results()[0]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(result1_input));
    EXPECT_EQ(result1_input->input_value(0).get_index(), 0u);

    auto result2_input = model->get_results()[1]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(result2_input));
    EXPECT_EQ(result2_input->input_value(0).get_index(), 1u);
}

// Test R10: Slice(Reshape(Tile(X))) -> Reshape(Tile(Slice(X)))
TEST(PropagateSliceTest, PropagateSliceThroughTileReshape) {
    // Build: Param[1024,2048] -> Tile(repeats=[128,1]) -> [131072,2048]
    //        -> Reshape -> [128,1024,2048] -> Slice(axis=1,-1:) -> [128,1,2048]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 2048});

    // Tile with repeats [128, 1]
    auto repeats = v0::Constant::create(element::i64, Shape{2}, {128, 1});
    auto tile = std::make_shared<v0::Tile>(param, repeats);

    // Reshape [131072, 2048] -> [128, 1024, 2048]
    auto pattern = v0::Constant::create(element::i64, Shape{3}, {128, 1024, 2048});
    auto reshape = std::make_shared<v1::Reshape>(tile, pattern, false);

    // Slice on axis=1: [128, 1024, 2048] -> [128, 1, 2048]
    auto slice = make_last_index_slice(reshape, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Verify initial shape
    ASSERT_EQ(result->get_input_shape(0), (Shape{128, 1, 2048}));

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Should have Reshape -> Tile -> Slice -> Param
    auto result_node = model->get_results()[0];
    auto reshape_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v1::Reshape>(reshape_node));

    auto tile_node = reshape_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Tile>(tile_node));

    auto slice_node = tile_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));

    // Verify final shape is still correct
    EXPECT_EQ(result_node->get_input_shape(0), (Shape{128, 1, 2048}));

    // Verify intermediate shapes
    // Slice: [1024, 2048] -> [1, 2048]
    EXPECT_EQ(slice_node->get_output_shape(0), (Shape{1, 2048}));
    // Tile: [1, 2048] -> [128, 2048]
    EXPECT_EQ(tile_node->get_output_shape(0), (Shape{128, 2048}));
    // Reshape: [128, 2048] -> [128, 1, 2048]
    EXPECT_EQ(reshape_node->get_output_shape(0), (Shape{128, 1, 2048}));
}

// Test R11: Slice(Unsqueeze(X)) -> Unsqueeze(Slice(X))
TEST(PropagateSliceTest, PropagateSliceThroughUnsqueeze) {
    // Build: Param[128,1024] -> Unsqueeze(axes=[2]) -> [128,1024,1] -> Slice(axis=1,-1:) -> [128,1,1]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{128, 1024});

    // Unsqueeze on axis=2
    auto axes = v0::Constant::create(element::i64, Shape{1}, {2});
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(param, axes);

    // Slice on axis=1: [128, 1024, 1] -> [128, 1, 1]
    auto slice = make_last_index_slice(unsqueeze, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Verify initial shape
    ASSERT_EQ(result->get_input_shape(0), (Shape{128, 1, 1}));

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Should have Unsqueeze -> Slice -> Param
    auto result_node = model->get_results()[0];
    auto unsqueeze_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v0::Unsqueeze>(unsqueeze_node));

    auto slice_node = unsqueeze_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));

    // Verify final shape is still correct
    EXPECT_EQ(result_node->get_input_shape(0), (Shape{128, 1, 1}));

    // Verify intermediate shapes
    // Slice: [128, 1024] -> [128, 1]
    EXPECT_EQ(slice_node->get_output_shape(0), (Shape{128, 1}));
    // Unsqueeze: [128, 1] -> [128, 1, 1]
    EXPECT_EQ(unsqueeze_node->get_output_shape(0), (Shape{128, 1, 1}));
}

// Test R12: Slice(ScatterElementsUpdate(...)) -> ScatterElementsUpdate(Slice(...))
TEST(PropagateSliceTest, PropagateSliceThroughScatterElementsUpdate) {
    // Build: data[1024,8], indices[1024,8], updates[1024,8]
    //        -> ScatterElementsUpdate(axis=1) -> [1024,8] -> Slice(axis=0,-1:) -> [1,8]
    auto data = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 8});
    auto indices = std::make_shared<v0::Parameter>(element::i32, Shape{1024, 8});
    auto updates = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 8});

    // ScatterElementsUpdate with axis=1 (v12 version)
    auto scatter_axis = v0::Constant::create(element::i64, Shape{}, {1});
    auto scatter = std::make_shared<v12::ScatterElementsUpdate>(data, indices, updates, scatter_axis);

    // Slice on axis=0: [1024, 8] -> [1, 8]
    auto slice = make_last_index_slice(scatter, 0);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{data, indices, updates});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Should have ScatterElementsUpdate with sliced inputs
    auto result_node = model->get_results()[0];
    auto scatter_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v12::ScatterElementsUpdate>(scatter_node));

    // All three inputs should be Slice nodes
    auto data_input = scatter_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(data_input));

    auto indices_input = scatter_node->input_value(1).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(indices_input));

    auto updates_input = scatter_node->input_value(2).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(updates_input));

    // Verify shapes: all sliced on axis=0, 1024 -> 1
    EXPECT_EQ(data_input->get_output_shape(0), (Shape{1, 8}));
    EXPECT_EQ(indices_input->get_output_shape(0), (Shape{1, 8}));
    EXPECT_EQ(updates_input->get_output_shape(0), (Shape{1, 8}));
}

// Test: shared single_consumer() guard blocks propagation through a multi-consumer parent
TEST(PropagateSliceTest, MultiConsumerBlocksPropagation) {
    // Build: Param -> Gelu (2 consumers) -> Slice on one branch
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto gelu = std::make_shared<v0::Gelu>(param);

    auto slice = make_last_index_slice(gelu, 1);

    // Create second consumer
    auto relu = std::make_shared<v0::Relu>(gelu);

    auto result1 = std::make_shared<v0::Result>(slice);
    auto result2 = std::make_shared<v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ResultVector{result1, result2}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Check: Slice should NOT have moved (Gelu has multiple consumers)
    auto result_node = model->get_results()[0];
    EXPECT_TRUE(is_type<v8::Slice>(result_node->input_value(0).get_node_shared_ptr()));
}

// Test R1 (Convert): Slice(Convert(X)) -> Convert(Slice(X))
TEST(PropagateSliceTest, ConvertUnary) {
    // Build: Param -> Convert -> Slice
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto convert = std::make_shared<v0::Convert>(param, element::f16);

    auto slice = make_last_index_slice(convert, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: Param -> Slice -> Convert
    auto result_node = model->get_results()[0];
    auto out_convert = result_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Convert>(out_convert));

    auto convert_input = out_convert->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(convert_input));

    auto slice_input = convert_input->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(slice_input));
}

// Test R13: Slice(Broadcast(X)) -> Broadcast(X) with adjusted target_shape
TEST(PropagateSliceTest, BroadcastSimplification) {
    // Build: Param([1,128]) -> Broadcast([1024,128]) -> Slice(axis=0, -1:) -> [1,128]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 128});
    auto target_shape = v0::Constant::create(element::i64, Shape{2}, {1024, 128});
    auto axes_mapping = v0::Constant::create(element::i64, Shape{2}, {0, 1});
    auto broadcast = std::make_shared<v3::Broadcast>(param, target_shape, axes_mapping);

    auto slice = make_last_index_slice(broadcast, 0);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: Broadcast should be adjusted to output [1,128] directly
    auto result_node = model->get_results()[0];
    auto out_node = result_node->input_value(0).get_node_shared_ptr();

    // Should be Broadcast directly, not Slice
    EXPECT_TRUE(is_type<v3::Broadcast>(out_node));
    EXPECT_EQ(out_node->get_output_shape(0), (Shape{1, 128}));
}

// Test R14: Remove no-op Slice nodes (input_shape == output_shape)
TEST(PropagateSliceTest, RemoveNoOpSlice) {
    // Build: Param([1024,128]) -> Slice(axis=0, 0:1024) [no-op] -> Gelu
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 128});

    // No-op slice: 0:1024 (same as input)
    auto no_op_slice = make_slice(param, 0, 0, 1024);

    auto gelu = std::make_shared<v0::Gelu>(no_op_slice);

    auto result = std::make_shared<v0::Result>(gelu);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: No-op slice should be removed, Gelu connected to param directly
    auto result_node = model->get_results()[0];
    auto out_gelu = result_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Gelu>(out_gelu));

    auto gelu_input = out_gelu->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(gelu_input));
}

// Test R15: Slice(TopK(X)[values]), Slice(TopK(X)[indices]) -> TopK(Slice(X))
TEST(PropagateSliceTest, PropagateSliceThroughTopK) {
    // Build: Param[1024,128] -> TopK(axis=1, k=8) -> values[1024,8] -> Slice[1,8]
    //                                                -> indices[1024,8] -> Slice[1,8]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 128});
    auto k = v0::Constant::create(element::i64, Shape{}, {8});
    auto topk = std::make_shared<v3::TopK>(param, k, 1, v3::TopK::Mode::MAX, v3::TopK::SortType::SORT_VALUES);

    // Slice on axis=0: [1024, 8] -> [1, 8]
    auto values_slice = make_slice(topk->output(0), 0, 0, 1);
    auto indices_slice = make_slice(topk->output(1), 0, 0, 1);

    auto result1 = std::make_shared<v0::Result>(values_slice);
    auto result2 = std::make_shared<v0::Result>(indices_slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result1, result2}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: TopK(Slice(param))
    // Result 1 should be TopK output 0 (values)
    auto result_node1 = model->get_results()[0];
    auto topk_node1 = result_node1->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v3::TopK>(topk_node1));
    EXPECT_EQ(topk_node1->get_output_shape(0), (Shape{1, 8}));

    // Result 2 should be TopK output 1 (indices)
    auto result_node2 = model->get_results()[1];
    auto topk_node2 = result_node2->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v3::TopK>(topk_node2));
    EXPECT_EQ(topk_node2->get_output_shape(1), (Shape{1, 8}));

    // Both should point to the same TopK node
    EXPECT_EQ(topk_node1.get(), topk_node2.get());

    // TopK input should be Slice of param
    auto slice_input = topk_node1->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_input));
    EXPECT_EQ(slice_input->get_output_shape(0), (Shape{1, 128}));

    auto param_input = slice_input->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(param_input));
}

// Test R16: Slice(Softmax(X, axis=A), axis=B) -> Softmax(Slice(X, axis=B), axis=A) where A != B
TEST(PropagateSliceTest, PropagateSliceThroughSoftmax) {
    // Build: Param[1024,128] -> Softmax(axis=1) -> [1024,128] -> Slice(axis=0) -> [1,128]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 128});
    auto softmax = std::make_shared<v8::Softmax>(param, 1);  // axis=1 (last dimension)

    // Slice on axis=0: [1024,128] -> [1,128]
    auto slice = make_slice(softmax, 0, 0, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: Softmax(Slice(param))
    auto result_node = model->get_results()[0];
    auto softmax_node = result_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Softmax>(softmax_node));
    EXPECT_EQ(softmax_node->get_output_shape(0), (Shape{1, 128}));

    // Softmax input should be Slice of param
    auto slice_input = softmax_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_input));
    EXPECT_EQ(slice_input->get_output_shape(0), (Shape{1, 128}));

    auto param_input = slice_input->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(param_input));
}

// Test R17: Slice(Concat(X1,X2,...,axis=A), axis=B) -> Concat(Slice(X1,axis=B),Slice(X2,axis=B),...,axis=A)
TEST(PropagateSliceTest, PropagateSliceThroughConcat) {
    // Build: Param1[1,32,1024,64] + Param2[1,32,1024,64] -> Concat(axis=3) -> [1,32,1024,128]
    //        -> Slice(axis=2) -> [1,32,1,128]
    auto param1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 32, 1024, 64});
    auto param2 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 32, 1024, 64});
    auto concat = std::make_shared<v0::Concat>(OutputVector{param1, param2}, 3);  // axis=3

    // Slice on axis=2: [1,32,1024,128] -> [1,32,1,128]
    auto slice = make_slice(concat, 2, 0, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param1, param2});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: Concat(Slice(param1), Slice(param2))
    auto result_node = model->get_results()[0];
    auto concat_node = result_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Concat>(concat_node));
    EXPECT_EQ(concat_node->get_output_shape(0), (Shape{1, 32, 1, 128}));

    // Both Concat inputs should be Slices
    auto input0 = concat_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(input0));
    EXPECT_EQ(input0->get_output_shape(0), (Shape{1, 32, 1, 64}));

    auto input1 = concat_node->input_value(1).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(input1));
    EXPECT_EQ(input1->get_output_shape(0), (Shape{1, 32, 1, 64}));

    // Slices should connect to params
    EXPECT_TRUE(is_type<v0::Parameter>(input0->input_value(0).get_node_shared_ptr()));
    EXPECT_TRUE(is_type<v0::Parameter>(input1->input_value(0).get_node_shared_ptr()));
}

// Test R18: Merge consecutive Slices on different axes
TEST(PropagateSliceTest, MergeConsecutiveSlicesOnDifferentAxes) {
    // Build: Param[1,32,1024,128] -> Slice(axis=3) -> [1,32,1024,64] -> Slice(axis=2) -> [1,32,1,64]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 32, 1024, 128});

    // First slice on axis=3: [1,32,1024,128] -> [1,32,1024,64]
    auto slice1 = make_slice(param, 3, 0, 64);

    // Second slice on axis=2: [1,32,1024,64] -> [1,32,1,64]
    auto slice2 = make_slice(slice1, 2, 0, 1);

    auto result = std::make_shared<v0::Result>(slice2);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: Single merged Slice(param) with axes=[2,3]
    auto result_node = model->get_results()[0];
    auto merged_slice = result_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(merged_slice));
    EXPECT_EQ(merged_slice->get_output_shape(0), (Shape{1, 32, 1, 64}));

    // Merged slice should directly connect to param
    auto slice_input = merged_slice->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(slice_input));

    // Verify that axes parameter contains both axes [2, 3]
    auto axes_node = merged_slice->input_value(4).get_node_shared_ptr();
    auto axes_const = std::dynamic_pointer_cast<v0::Constant>(axes_node);
    ASSERT_TRUE(axes_const != nullptr);
    auto axes_vec = axes_const->cast_vector<int64_t>();
    EXPECT_EQ(axes_vec.size(), 2);
    // Note: axes might be in sorted order
    EXPECT_TRUE((axes_vec[0] == 2 && axes_vec[1] == 3) || (axes_vec[0] == 3 && axes_vec[1] == 2));
}

// Test R19: Extract common slice axes before Transpose
TEST(PropagateSliceTest, ExtractCommonSliceBeforeTranspose) {
    // Build: Transpose([1,1024,32,128] -> [1,32,1024,128])
    //        -> Slice1(axes=[2,3]): -> [1,32,1,64]     (axis=2: 0:1, axis=3: 0:64)
    //        -> Slice2(axes=[2,3]): -> [1,32,1,64]     (axis=2: 0:1, axis=3: 64:128)
    //        -> Slice3(axes=[2]):   -> [1,32,1,128]    (axis=2: 0:1 only)
    // Common axis: 2 (all slices: 0:1:1)
    // Expected: Slice(axis=1: 0:1:1) -> Transpose -> residual slices on axis=3

    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 32, 128});
    auto order = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});  // [0,1,2,3] -> [0,2,1,3]
    auto transpose = std::make_shared<v1::Transpose>(param, order);

    // Slice1: axes=[2,3], start=[0,0], stop=[1,64]
    auto slice1 = make_multi_axis_slice(transpose, {2, 3}, {0, 0}, {1, 64});

    // Slice2: axes=[2,3], start=[0,64], stop=[1,128]
    auto slice2 = make_multi_axis_slice(transpose, {2, 3}, {0, 64}, {1, 128});

    // Slice3: axes=[2], start=[0], stop=[1]
    auto slice3 = make_slice(transpose, 2, 0, 1);

    auto result1 = std::make_shared<v0::Result>(slice1);
    auto result2 = std::make_shared<v0::Result>(slice2);
    auto result3 = std::make_shared<v0::Result>(slice3);
    auto model = std::make_shared<ov::Model>(ResultVector{result1, result2, result3}, ParameterVector{param});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected structure: Slice(axis=1) -> Transpose -> [residual_slice1, residual_slice2, result3]
    // Find the Transpose in the graph
    std::shared_ptr<v1::Transpose> new_transpose;
    for (const auto& node : model->get_ops()) {
        auto trans = std::dynamic_pointer_cast<v1::Transpose>(node);
        if (trans) {
            new_transpose = trans;
            break;
        }
    }
    ASSERT_TRUE(new_transpose != nullptr);

    // Transpose input should be a Slice on param
    auto transpose_input = new_transpose->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(transpose_input));

    // That slice should connect to param
    auto slice_input = transpose_input->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(slice_input));

    // Transpose output shape should be [1,32,1,128]
    EXPECT_EQ(new_transpose->get_output_shape(0), (Shape{1, 32, 1, 128}));

    // Result1 should have a residual slice (axis=3 only)
    auto result1_input = model->get_results()[0]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(result1_input));
    EXPECT_EQ(result1_input->get_output_shape(0), (Shape{1, 32, 1, 64}));

    // Result2 should have a residual slice (axis=3 only)
    auto result2_input = model->get_results()[1]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(result2_input));
    EXPECT_EQ(result2_input->get_output_shape(0), (Shape{1, 32, 1, 64}));

    // Result3 should connect directly to Transpose (no residual slice)
    auto result3_input = model->get_results()[2]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v1::Transpose>(result3_input));
    EXPECT_EQ(result3_input->get_output_shape(0), (Shape{1, 32, 1, 128}));
}

// Test R20: Extract common slice axes before Binary op
TEST(PropagateSliceTest, ExtractCommonSliceBeforeBinary) {
    // Build: Multiply([1,1,1,512], [1,1024,8,512]) -> [1,1024,8,512]
    //        -> Slice1(axes=[1,3]): -> [1,1,8,512]     (axis=1: 0:1, axis=3: 0:512)
    //        -> Slice2(axes=[1,3]): -> [1,1,8,256]     (axis=1: 0:1, axis=3: 0:256)
    //        -> Slice3(axes=[1,3]): -> [1,1,8,256]     (axis=1: 0:1, axis=3: 256:512)
    // Common axis: 1 (all slices: 0:1:1)
    // Expected: Multiply([1,1,1,512], Slice([1,1024,8,512]->[1,1,8,512])) -> [1,1,8,512]
    //           -> residual slices on axis=3

    auto param1 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1, 1, 512});
    auto param2 = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 8, 512});
    auto multiply = std::make_shared<v1::Multiply>(param1, param2);

    // Slice1/2/3 are constructed inline (not kept as named locals) so that once the pass
    // rewires the Results away from them, the old Slice nodes are actually destroyed.
    // Otherwise they'd keep dangling input edges into Multiply alive, making it look like
    // it still has multiple consumers and blocking further propagation.
    auto result1 = std::make_shared<v0::Result>(make_multi_axis_slice(multiply, {1, 3}, {0, 0}, {1, 512}));
    auto result2 = std::make_shared<v0::Result>(make_multi_axis_slice(multiply, {1, 3}, {0, 0}, {1, 256}));
    auto result3 = std::make_shared<v0::Result>(make_multi_axis_slice(multiply, {1, 3}, {0, 256}, {1, 512}));
    auto model = std::make_shared<ov::Model>(ResultVector{result1, result2, result3}, ParameterVector{param1, param2});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected structure: Multiply([1,1,1,512], Slice(param2)) -> residual slices
    // Find the Multiply in the graph
    std::shared_ptr<v1::Multiply> new_multiply;
    for (const auto& node : model->get_ops()) {
        auto mult = std::dynamic_pointer_cast<v1::Multiply>(node);
        if (mult) {
            new_multiply = mult;
            break;
        }
    }
    ASSERT_TRUE(new_multiply != nullptr);

    // Multiply's second input should be a Slice on param2
    auto multiply_input2 = new_multiply->input_value(1).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(multiply_input2));

    // That slice should connect to param2
    auto slice_input = multiply_input2->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(slice_input));

    // Multiply output shape should be [1,1,8,512]
    EXPECT_EQ(new_multiply->get_output_shape(0), (Shape{1, 1, 8, 512}));

    // Result1 should connect directly to Multiply (no residual slice, axis=3 is full range)
    auto result1_input = model->get_results()[0]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v1::Multiply>(result1_input));
    EXPECT_EQ(result1_input->get_output_shape(0), (Shape{1, 1, 8, 512}));

    // Result2 should have a residual slice (axis=3 only)
    auto result2_input = model->get_results()[1]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(result2_input));
    EXPECT_EQ(result2_input->get_output_shape(0), (Shape{1, 1, 8, 256}));

    // Result3 should have a residual slice (axis=3 only)
    auto result3_input = model->get_results()[2]->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(result3_input));
    EXPECT_EQ(result3_input->get_output_shape(0), (Shape{1, 1, 8, 256}));
}

// Test R17a: Slice(Gather(X, indices, axis=2), axis=1) -> Gather(Slice(X, axis=1), indices, axis=2)
TEST(PropagateSliceTest, PropagateSliceThroughGather) {
    // Build: Param[1,1024,35,256] -> Gather(axis=2, indices=constant) -> [1,1024,256]
    //        -> Slice(axis=1, 1024->1) -> [1,1,256]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 35, 256});

    // Gather on axis=2, select one element (index 0)
    auto indices = v0::Constant::create(element::i64, Shape{}, {0});
    auto gather_axis = v0::Constant::create(element::i64, Shape{}, {2});
    auto gather = std::make_shared<v8::Gather>(param, indices, gather_axis);

    // Slice on axis=1 (1024->1)
    auto slice = make_slice(gather, 1, 0, 1);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Apply transformation
    apply_propagate_slice_up(model);

    // Expected: Parameter -> Slice(axis=1) -> Gather(axis=2) -> Result

    // Find the Result's input
    auto result_input = model->get_results()[0]->input_value(0).get_node_shared_ptr();

    // Should be Gather
    EXPECT_TRUE(is_type<v8::Gather>(result_input));
    auto new_gather = std::dynamic_pointer_cast<v8::Gather>(result_input);
    EXPECT_EQ(new_gather->get_output_shape(0), (Shape{1, 1, 256}));

    // Gather's input should be Slice
    auto gather_input = new_gather->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(gather_input));
    auto new_slice = std::dynamic_pointer_cast<v8::Slice>(gather_input);
    EXPECT_EQ(new_slice->get_output_shape(0), (Shape{1, 1, 35, 256}));

    // Slice's input should be Parameter
    auto slice_input = new_slice->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v0::Parameter>(slice_input));
}

// Test R3: Slice(SDPA(Q,K,V), axis=seq) -> SDPA(Slice(Q,axis=seq), K, V)
TEST(PropagateSliceTest, PropagateSliceThroughSDPA) {
    // Build: Q,K,V[1,8,1024,64] -> SDPA(causal=false) -> [1,8,1024,64]
    //        -> Slice(axis=2, seq: 1024->1) -> [1,8,1,64]
    auto q = std::make_shared<v0::Parameter>(element::f32, Shape{1, 8, 1024, 64});
    auto k = std::make_shared<v0::Parameter>(element::f32, Shape{1, 8, 1024, 64});
    auto v = std::make_shared<v0::Parameter>(element::f32, Shape{1, 8, 1024, 64});
    auto sdpa = std::make_shared<v13::ScaledDotProductAttention>(q, k, v, false);

    // Slice on axis=2 (query sequence axis): 1024 -> 1
    auto slice = make_last_index_slice(sdpa, 2);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{q, k, v});

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: SDPA(Slice(Q), K, V) -- only Q is sliced, K/V are unaffected
    auto result_node = model->get_results()[0];
    auto sdpa_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v13::ScaledDotProductAttention>(sdpa_node));
    EXPECT_EQ(sdpa_node->get_output_shape(0), (Shape{1, 8, 1, 64}));

    auto q_input = sdpa_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(q_input));
    EXPECT_EQ(q_input->get_output_shape(0), (Shape{1, 8, 1, 64}));
    EXPECT_TRUE(is_type<v0::Parameter>(q_input->input_value(0).get_node_shared_ptr()));

    // K and V should remain the original (unsliced) Parameters
    EXPECT_TRUE(is_type<v0::Parameter>(sdpa_node->input_value(1).get_node_shared_ptr()));
    EXPECT_TRUE(is_type<v0::Parameter>(sdpa_node->input_value(2).get_node_shared_ptr()));
}

// Test R6 (non-squeeze-like): Slice(Reshape(X)) where Reshape genuinely splits a dimension
// (not just inserts/removes size-1 dims), so the pass must update the Reshape's constant
// pattern directly instead of inserting an Unsqueeze.
TEST(PropagateSliceTest, PropagateSliceThroughReshape_UpdatePattern) {
    // Build: Param[1024,6] -> Reshape([1024,2,3]) -> Slice(axis=0, 1024->1) -> [1,2,3]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1024, 6});
    auto pattern = v0::Constant::create(element::i64, Shape{3}, {1024, 2, 3});
    auto reshape = std::make_shared<v1::Reshape>(param, pattern, false);

    auto slice = make_last_index_slice(reshape, 0);

    auto result = std::make_shared<v0::Result>(slice);
    auto model = std::make_shared<ov::Model>(ResultVector{result}, ParameterVector{param});

    // Verify initial shape
    ASSERT_EQ(result->get_input_shape(0), (Shape{1, 2, 3}));

    // Apply pass
    apply_propagate_slice_up(model);

    // Expected: Reshape(Slice(param)) with an updated constant pattern (still a Reshape, not Unsqueeze)
    auto result_node = model->get_results()[0];
    auto reshape_node = result_node->input_value(0).get_node_shared_ptr();
    ASSERT_TRUE(is_type<v1::Reshape>(reshape_node));
    EXPECT_EQ(reshape_node->get_output_shape(0), (Shape{1, 2, 3}));

    auto slice_node = reshape_node->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(slice_node));
    EXPECT_EQ(slice_node->get_output_shape(0), (Shape{1, 6}));
    EXPECT_TRUE(is_type<v0::Parameter>(slice_node->input_value(0).get_node_shared_ptr()));

    // Reshape's pattern constant should reflect the sliced first dimension
    auto pattern_node = std::dynamic_pointer_cast<v0::Constant>(reshape_node->input_value(1).get_node_shared_ptr());
    ASSERT_TRUE(pattern_node != nullptr);
    EXPECT_EQ(pattern_node->cast_vector<int64_t>(), (std::vector<int64_t>{1, 2, 3}));
}

// Test R8: Slice(VariadicSplit(X)[i]) for all i -> VariadicSplit(Slice(X))
TEST(PropagateSliceTest, PropagateSliceThroughVariadicSplit) {
    // Build: Param[1,1024,3072] -> VariadicSplit(axis=2, split_lengths=[1024,1024,1024])
    //        -> 3x [1,1024,1024], each followed by an identical Slice(axis=1, 0:1) -> [1,1,1024]
    auto param = std::make_shared<v0::Parameter>(element::f32, Shape{1, 1024, 3072});
    auto split_axis = v0::Constant::create(element::i64, Shape{}, {2});
    auto split_lengths = v0::Constant::create(element::i64, Shape{3}, {1024, 1024, 1024});
    auto vsplit = std::make_shared<v1::VariadicSplit>(param, split_axis, split_lengths);

    auto slice0 = make_slice(vsplit->output(0), 1, 0, 1);
    auto slice1 = make_slice(vsplit->output(1), 1, 0, 1);
    auto slice2 = make_slice(vsplit->output(2), 1, 0, 1);

    auto result0 = std::make_shared<v0::Result>(slice0);
    auto result1 = std::make_shared<v0::Result>(slice1);
    auto result2 = std::make_shared<v0::Result>(slice2);
    auto model = std::make_shared<ov::Model>(ResultVector{result0, result1, result2}, ParameterVector{param});

    // Before: 3 Slice nodes (one per VariadicSplit output)
    ASSERT_EQ(count_slice_nodes(model), 3);

    // Apply pass
    apply_propagate_slice_up(model);

    // After: a single common Slice moved before VariadicSplit; per-output Slices are gone
    EXPECT_EQ(count_slice_nodes(model), 1);

    // Each Result should now connect directly to a VariadicSplit output
    std::shared_ptr<v1::VariadicSplit> new_vsplit;
    for (size_t i = 0; i < 3; ++i) {
        auto result_input = model->get_results()[i]->input_value(0).get_node_shared_ptr();
        EXPECT_TRUE(is_type<v1::VariadicSplit>(result_input));
        EXPECT_EQ(result_input->get_output_shape(i), (Shape{1, 1, 1024}));
        new_vsplit = std::dynamic_pointer_cast<v1::VariadicSplit>(result_input);
    }

    // VariadicSplit's data input should be the propagated common Slice
    ASSERT_TRUE(new_vsplit != nullptr);
    auto vsplit_input = new_vsplit->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(is_type<v8::Slice>(vsplit_input));
    EXPECT_EQ(vsplit_input->get_output_shape(0), (Shape{1, 1, 3072}));
    EXPECT_TRUE(is_type<v0::Parameter>(vsplit_input->input_value(0).get_node_shared_ptr()));
}
