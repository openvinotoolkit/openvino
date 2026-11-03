// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "projector_adapter.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {
namespace {
using namespace ov::op;
using Value = ov::Output<ov::Node>;
using Models = ProjectorModels;

Value range(const Value& count) {
    auto axis = v0::Constant::create(ov::element::i64, {1}, {0});
    auto start = v0::Constant::create(ov::element::i64, {}, {0});
    auto step = v0::Constant::create(ov::element::i64, {}, {1});
    auto stop = std::make_shared<v0::Squeeze>(count, axis);
    auto indices = std::make_shared<v4::Range>(start, stop, step, ov::element::i64);
    return indices;
}

std::shared_ptr<v0::Parameter> parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto candidate = find_parameter(model, name);
    OPENVINO_ASSERT(candidate, "[GGUF] mmproj vision encoder has no input ", name);
    return candidate;
}

Value flat_index(const Value& value) {
    auto index_shape = v0::Constant::create(ov::element::i64, {4}, {1, 1, 1, -1});
    return std::make_shared<v1::Reshape>(value, index_shape, false);
}

void require_square_fits(int64_t value, int64_t factor, const char* key) {
    OPENVINO_ASSERT(value <= std::numeric_limits<int64_t>::max() / factor / value,
                    "[GGUF] out-of-range mmproj metadata ",
                    key);
}

void replace_input(const std::shared_ptr<ov::Model>& model, const std::string& name, Value value) {
    auto source = parameter(model, name);
    if (value.get_element_type() != source->get_element_type()) {
        auto converted = std::make_shared<v0::Convert>(value, source->get_element_type());
        value = converted;
    }
    source->output(0).replace(value);
    value.get_tensor().set_names({});
    model->remove_parameter(source);
}

void add_inputs(const std::shared_ptr<ov::Model>& model,
                const std::vector<std::pair<std::string, std::shared_ptr<v0::Parameter>>>& inputs) {
    for (const auto& [name, input] : inputs) {
        name_output(input, name);
        model->add_parameters({input});
    }
    model->validate_nodes_and_infer_types();
}

// The translator names a GGUF tensor "<tensor>_<tensor>".
Value weight(const std::shared_ptr<ov::Model>& model, const std::string& tensor) {
    const auto name = tensor + "_" + tensor;
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == name)
            return node->output(0);
    }
    return {};
}

Value require_weight(const std::shared_ptr<ov::Model>& model, const std::string& tensor) {
    auto value = weight(model, tensor);
    OPENVINO_ASSERT(value.get_node(), "[GGUF] mmproj vision encoder has no ", tensor);
    return value;
}

Value image_from_patches(const Value& patches,
                         const Value& rows,
                         const Value& cols,
                         int64_t patch,
                         bool channels_last) {
    auto p = v0::Constant::create(ov::element::i64, {1}, {patch});
    auto c = v0::Constant::create(ov::element::i64, {1}, {3});
    auto grid = channels_last ? ov::OutputVector{rows, cols, p, p, c} : ov::OutputVector{rows, cols, c, p, p};
    auto grid_shape = std::make_shared<v0::Concat>(grid, 0);
    auto patch_grid = std::make_shared<v1::Reshape>(patches, grid_shape, false);
    auto order =
        v0::Constant::create(ov::element::i64,
                             {5},
                             channels_last ? std::vector<int64_t>{4, 0, 2, 1, 3} : std::vector<int64_t>{2, 0, 3, 1, 4});
    auto image_grid = std::make_shared<v1::Transpose>(patch_grid, order);
    auto batch_channels = v0::Constant::create(ov::element::i64, {2}, {1, 3});
    auto patch_size = v0::Constant::create(ov::element::i64, {}, {patch});
    auto height = std::make_shared<v1::Multiply>(rows, patch_size);
    auto width = std::make_shared<v1::Multiply>(cols, patch_size);
    auto image_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_channels, height, width}, 0);
    auto image = std::make_shared<v1::Reshape>(image_grid, image_shape, false);
    return image;
}

void squeeze_output(const std::shared_ptr<ov::Model>& model) {
    auto result = model->get_results().front();
    auto axis = v0::Constant::create(ov::element::i64, {1}, {0});
    auto features = std::make_shared<v0::Squeeze>(result->input_value(0), axis);
    features->output(0).get_tensor().set_names({"last_hidden_state"});
    result->input(0).replace_source_output(features);
}

// Patches fill the leading rows in raster order; padding rows have position -1.
void padded_patch_layout(const std::shared_ptr<ov::Model>& model, int64_t patch) {
    require_square_fits(patch, 3, "vision.patch_size");
    auto pixels = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3 * patch * patch});
    auto positions = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1, 2});
    auto zero_axis = v0::Constant::create(ov::element::i64, {1}, {0});
    auto one_step = v0::Constant::create(ov::element::i64, {1}, {1});
    auto zero = v0::Constant::create(ov::element::i64, {}, {0});
    auto one = v0::Constant::create(ov::element::i64, {}, {1});
    auto flat = std::make_shared<v0::Squeeze>(positions, zero_axis);
    auto x = std::make_shared<v8::Gather>(flat, zero, one);
    auto y = std::make_shared<v8::Gather>(flat, one, one);
    auto nonnegative = std::make_shared<v1::GreaterEqual>(x, zero);
    auto valid_rows = std::make_shared<v0::Convert>(nonnegative, ov::element::i64);
    auto count = std::make_shared<v1::ReduceSum>(valid_rows, zero_axis, true);
    auto maximum_x = std::make_shared<v1::ReduceMax>(x, zero_axis, true);
    auto cols = std::make_shared<v1::Add>(maximum_x, one);
    auto rows = std::make_shared<v1::Divide>(count, cols);
    auto flat_pixels = std::make_shared<v0::Squeeze>(pixels, zero_axis);
    auto valid_pixels = std::make_shared<v8::Slice>(flat_pixels, zero_axis, count, one_step, zero_axis);
    auto valid_x = std::make_shared<v8::Slice>(x, zero_axis, count, one_step, zero_axis);
    auto valid_y = std::make_shared<v8::Slice>(y, zero_axis, count, one_step, zero_axis);
    auto image = image_from_patches(valid_pixels, rows, cols, patch, true);
    auto position_x = flat_index(valid_x);
    auto position_y = flat_index(valid_y);
    replace_input(model, "pixel_values", image);
    replace_input(model, "position_x", position_x);
    replace_input(model, "position_y", position_y);
    add_inputs(model, {{"pixel_values", pixels}, {"image_position_ids", positions}});
}

void windowed_patch_layout(const std::shared_ptr<ov::Model>& model, int64_t patch, int64_t window, int64_t merge) {
    require_square_fits(patch, 3, "vision.patch_size");
    require_square_fits(window, 1, "vision.window_size");
    require_square_fits(merge, 1, "vision.merge");
    auto pixels = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3 * patch * patch});
    auto grid = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3});
    auto zero_axis = v0::Constant::create(ov::element::i64, {1}, {0});
    auto row_index = v0::Constant::create(ov::element::i64, {1}, {1});
    auto col_index = v0::Constant::create(ov::element::i64, {1}, {2});
    auto zero = v0::Constant::create(ov::element::i64, {}, {0});
    auto one = v0::Constant::create(ov::element::i64, {}, {1});
    auto thw = std::make_shared<v0::Squeeze>(grid, zero_axis);
    auto rows = std::make_shared<v8::Gather>(thw, row_index, zero);
    auto cols = std::make_shared<v8::Gather>(thw, col_index, zero);
    auto image = image_from_patches(pixels, rows, cols, patch, false);
    replace_input(model, "pixel_values", image);

    // Slots outside the image repeat the window's first patch and are masked out as keys.
    const int64_t slots = window * window;
    auto window_size = v0::Constant::create(ov::element::i64, {}, {window});
    auto window_padding = v0::Constant::create(ov::element::i64, {}, {window - 1});
    auto window_slots = v0::Constant::create(ov::element::i64, {}, {slots});
    auto padded_cols = std::make_shared<v1::Add>(cols, window_padding);
    auto window_cols = std::make_shared<v1::Divide>(padded_cols, window_size);
    auto padded_rows = std::make_shared<v1::Add>(rows, window_padding);
    auto window_rows = std::make_shared<v1::Divide>(padded_rows, window_size);
    auto window_count = std::make_shared<v1::Multiply>(window_rows, window_cols);
    auto slot_count = std::make_shared<v1::Multiply>(window_count, window_slots);
    auto slot = range(slot_count);
    auto index_in_window = std::make_shared<v1::FloorMod>(slot, window_slots);
    auto window_index = std::make_shared<v1::Divide>(slot, window_slots);
    auto window_row = std::make_shared<v1::Divide>(window_index, window_cols);
    auto window_col = std::make_shared<v1::FloorMod>(window_index, window_cols);
    auto top = std::make_shared<v1::Multiply>(window_row, window_size);
    auto left = std::make_shared<v1::Multiply>(window_col, window_size);
    auto row_in_window = std::make_shared<v1::Divide>(index_in_window, window_size);
    auto col_in_window = std::make_shared<v1::FloorMod>(index_in_window, window_size);
    auto y = std::make_shared<v1::Add>(top, row_in_window);
    auto x = std::make_shared<v1::Add>(left, col_in_window);
    auto inside_rows = std::make_shared<v1::Less>(y, rows);
    auto inside_cols = std::make_shared<v1::Less>(x, cols);
    auto inside = std::make_shared<v1::LogicalAnd>(inside_rows, inside_cols);
    auto row_offset = std::make_shared<v1::Multiply>(y, cols);
    auto patch_index = std::make_shared<v1::Add>(row_offset, x);
    auto top_offset = std::make_shared<v1::Multiply>(top, cols);
    auto first_patch = std::make_shared<v1::Add>(top_offset, left);
    auto index = std::make_shared<v1::Select>(inside, patch_index, first_patch);
    auto patch_indices = flat_index(index);
    auto patch_x = std::make_shared<v1::FloorMod>(index, cols);
    auto patch_y = std::make_shared<v1::Divide>(index, cols);
    auto offset_x = std::make_shared<v1::Add>(patch_x, one);
    auto offset_y = std::make_shared<v1::Add>(patch_y, one);
    auto position_x = flat_index(offset_x);
    auto position_y = flat_index(offset_y);
    replace_input(model, "patch_indices", patch_indices);
    replace_input(model, "position_x", position_x);
    replace_input(model, "position_y", position_y);
    auto unmasked = v0::Constant::create(ov::element::f32, {}, {0.f});
    auto masked = v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<float>::infinity()});
    auto key_mask = std::make_shared<v1::Select>(inside, unmasked, masked);
    auto mask_shape = v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{-1, 1, 1, slots});
    auto window_mask = std::make_shared<v1::Reshape>(key_mask, mask_shape, false);
    replace_input(model, "window_mask", window_mask);

    auto patch_count = std::make_shared<v1::Multiply>(rows, cols);
    auto raster = range(patch_count);
    auto r = std::make_shared<v1::Divide>(raster, cols);
    auto c = std::make_shared<v1::FloorMod>(raster, cols);
    auto raster_window_row = std::make_shared<v1::Divide>(r, window_size);
    auto raster_window_col = std::make_shared<v1::Divide>(c, window_size);
    auto raster_window_offset = std::make_shared<v1::Multiply>(raster_window_row, window_cols);
    auto raster_window_index = std::make_shared<v1::Add>(raster_window_offset, raster_window_col);
    auto raster_slot_offset = std::make_shared<v1::Multiply>(raster_window_index, window_slots);
    auto raster_row_in_window = std::make_shared<v1::FloorMod>(r, window_size);
    auto raster_col_in_window = std::make_shared<v1::FloorMod>(c, window_size);
    auto raster_row_offset = std::make_shared<v1::Multiply>(raster_row_in_window, window_size);
    auto raster_index_in_window = std::make_shared<v1::Add>(raster_row_offset, raster_col_in_window);
    auto slot_of = std::make_shared<v1::Add>(raster_slot_offset, raster_index_in_window);
    auto output_indices = flat_index(slot_of);
    replace_input(model, "output_indices", output_indices);
    auto merge_size = v0::Constant::create(ov::element::i64, {}, {merge});
    auto merge_slots = v0::Constant::create(ov::element::i64, {}, {merge * merge});
    auto block = std::make_shared<v1::Divide>(raster, merge_slots);
    auto within = std::make_shared<v1::FloorMod>(raster, merge_slots);
    auto block_cols = std::make_shared<v1::Divide>(cols, merge_size);
    auto block_row = std::make_shared<v1::Divide>(block, block_cols);
    auto block_col = std::make_shared<v1::FloorMod>(block, block_cols);
    auto block_top = std::make_shared<v1::Multiply>(block_row, merge_size);
    auto block_left = std::make_shared<v1::Multiply>(block_col, merge_size);
    auto row_in_block = std::make_shared<v1::Divide>(within, merge_size);
    auto col_in_block = std::make_shared<v1::FloorMod>(within, merge_size);
    auto merged_row = std::make_shared<v1::Add>(block_top, row_in_block);
    auto merged_col = std::make_shared<v1::Add>(block_left, col_in_block);
    auto merged_row_offset = std::make_shared<v1::Multiply>(merged_row, cols);
    auto merged = std::make_shared<v1::Add>(merged_row_offset, merged_col);
    auto merge_indices = flat_index(merged);
    replace_input(model, "merge_indices", merge_indices);
    add_inputs(model, {{"pixel_values", pixels}, {"image_grid_thw", grid}});
    squeeze_output(model);
}

std::shared_ptr<ov::Model> single_output_model(const Value& output,
                                               const std::string& name,
                                               const ov::ParameterVector& inputs) {
    auto result = std::make_shared<v0::Result>(output);
    output.get_tensor().set_names({name});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, inputs);
    return model->clone();
}

Models split_merger_layout(const std::shared_ptr<ov::Model>& model) {
    Models models;
    // HF flattens patches as (channel, frame, y, x).
    auto first_weight = require_weight(model, "v.patch_embd.weight");
    auto second_weight = require_weight(model, "v.patch_embd.weight.1");
    auto frame_axis = v0::Constant::create(ov::element::i64, {1}, {2});
    auto first_frame = std::make_shared<v0::Unsqueeze>(first_weight, frame_axis);
    auto second_frame = std::make_shared<v0::Unsqueeze>(second_weight, frame_axis);
    auto kernel = std::make_shared<v0::Concat>(ov::OutputVector{first_frame, second_frame}, 2);
    const auto& kernel_shape = kernel->get_output_partial_shape(0);
    const auto width = kernel_shape[0].get_length();
    const auto patch_dim = ov::shape_size(kernel_shape.get_shape()) / width;
    auto patches = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, int64_t(patch_dim)});
    name_output(patches, "hidden_states");
    auto kernel_shape_2d = v0::Constant::create(ov::element::i64, {2}, {width, int64_t(patch_dim)});
    auto kernel_2d = std::make_shared<v1::Reshape>(kernel, kernel_shape_2d, false);
    auto projection = std::make_shared<v0::MatMul>(patches, kernel_2d, false, true);
    Value embeddings = projection;
    if (auto bias = weight(model, "v.patch_embd.bias"); bias.get_node()) {
        auto biased = std::make_shared<v1::Add>(embeddings, bias);
        embeddings = biased;
    }
    models["vision_embeddings"] = single_output_model(embeddings, "last_hidden_state", {patches});

    auto table_indices = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{4, -1});
    name_output(table_indices, "input");
    auto position_weight = require_weight(model, "v.position_embd.weight");
    auto zero = v0::Constant::create(ov::element::i64, {}, {0});
    auto table = std::make_shared<v8::Gather>(position_weight, table_indices, zero);
    models["vision_embeddings_pos"] = single_output_model(table, "last_hidden_state", {table_indices});

    auto norm_weight = require_weight(model, "v.blk.0.ln1.weight");
    const auto consumers = norm_weight.get_target_inputs();
    OPENVINO_ASSERT(consumers.size() == 1, "[GGUF] expected one consumer of v.blk.0.ln1.weight");
    auto scale = consumers.begin()->get_node();
    OPENVINO_ASSERT(ov::is_type<v1::Multiply>(scale), "[GGUF] unexpected consumer of v.blk.0.ln1.weight");
    auto normalized = ov::as_type_ptr<v6::MVN>(
        (scale->input_value(0) == norm_weight ? scale->input_value(1) : scale->input_value(0)).get_node_shared_ptr());
    OPENVINO_ASSERT(normalized, "[GGUF] expected LayerNorm before v.blk.0.ln1.weight");
    auto hidden_states = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, width});
    auto batch_axes = v0::Constant::create(ov::element::i64, {2}, {0, 1});
    auto hidden_states_4d = std::make_shared<v0::Unsqueeze>(hidden_states, batch_axes);
    // Replace the activation shared by LayerNorm and its residual path.
    normalized->input_value(0).replace(hidden_states_4d);
    model->remove_parameter(parameter(model, "pixel_values"));
    model->remove_parameter(parameter(model, "patch_indices"));

    // The first column of each rotary half has frequency 1, so it holds the row or column.
    const auto width_metadata = mmproj_int_metadata(model, "clip.vision.embedding_length");
    const auto heads = mmproj_int_metadata(model, "clip.vision.attention.head_count");
    OPENVINO_ASSERT(width_metadata == width && width_metadata % heads == 0 && width_metadata / heads % 4 == 0,
                    "[GGUF] incompatible mmproj metadata clip.vision.embedding_length and "
                    "clip.vision.attention.head_count");
    const auto head = width_metadata / heads;
    auto rotary = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, head / 2});
    auto col_index = v0::Constant::create(ov::element::i64, {}, {head / 4});
    auto one = v0::Constant::create(ov::element::i64, {}, {1});
    auto row = std::make_shared<v8::Gather>(rotary, zero, one);
    auto col = std::make_shared<v8::Gather>(rotary, col_index, one);
    auto coordinates = std::make_shared<v0::Concat>(ov::OutputVector{row, col, row, col}, 0);
    auto positions = std::make_shared<v5::Round>(coordinates, v5::Round::RoundMode::HALF_TO_EVEN);
    auto position_ids = flat_index(positions);
    replace_input(model, "position_ids", position_ids);

    auto mask = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, -1});
    auto zero_axis = v0::Constant::create(ov::element::i64, {1}, {0});
    auto mask4 = std::make_shared<v0::Unsqueeze>(mask, zero_axis);
    for (const auto& node : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<v13::ScaledDotProductAttention>(node)) {
            OPENVINO_ASSERT(sdpa->get_input_size() == 3, "[GGUF] split-merger vision attention already has a mask");
            auto masked = std::make_shared<v13::ScaledDotProductAttention>(sdpa->input_value(0),
                                                                           sdpa->input_value(1),
                                                                           sdpa->input_value(2),
                                                                           mask4,
                                                                           false);
            masked->set_friendly_name(sdpa->get_friendly_name());
            ov::replace_node(sdpa, masked);
        }
    }

    const auto results = model->get_results();
    squeeze_output(model);
    if (results.size() > 1) {
        ov::OutputVector levels;
        for (size_t i = 1; i < results.size(); ++i) {
            levels.push_back(results[i]->input_value(0));
            model->remove_result(results[i]);
        }
        auto deepstack = std::make_shared<v0::Concat>(levels, 0);
        deepstack->output(0).get_tensor().set_names({"deepstack_feature_lists"});
        auto deepstack_result = std::make_shared<v0::Result>(deepstack);
        model->add_results({deepstack_result});
    }
    add_inputs(model, {{"hidden_states", hidden_states}, {"attention_mask", mask}, {"rotary_pos_emb", rotary}});
    models["vision_embeddings_merger"] = model;
    return models;
}
void packed_patch_layout(const std::shared_ptr<ov::Model>& model, int64_t patch, int64_t merge) {
    require_square_fits(patch, 3, "vision.patch_size");
    OPENVINO_ASSERT(merge == 2 || merge == 4, "[GGUF] invalid packed-patch merge size");
    auto pixels = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, patch, -1});
    auto positions = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    auto window_index = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    auto merge_index = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    auto grid = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto zero = v0::Constant::create(ov::element::i64, {}, {0});
    auto row_index = v0::Constant::create(ov::element::i64, {1}, {0});
    auto col_index = v0::Constant::create(ov::element::i64, {1}, {1});
    auto rows = std::make_shared<v8::Gather>(grid, row_index, zero);
    auto cols = std::make_shared<v8::Gather>(grid, col_index, zero);
    auto strip_shape = v0::Constant::create(ov::element::i64, {5}, std::vector<int64_t>{1, 3, patch, -1, patch});
    auto strip = std::make_shared<v1::Reshape>(pixels, strip_shape, false);
    auto strip_order = v0::Constant::create(ov::element::i64, {5}, {0, 3, 1, 2, 4});
    auto reordered = std::make_shared<v1::Transpose>(strip, strip_order);
    auto packed_shape = v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, 3 * patch * patch});
    auto packed = std::make_shared<v1::Reshape>(reordered, packed_shape, false);
    auto image = image_from_patches(packed, rows, cols, patch, false);
    replace_input(model, "pixel_values", image);
    auto flat_positions = flat_index(positions);
    replace_input(model, "position_ids", flat_positions);
    auto merge_count = std::make_shared<v3::ShapeOf>(merge_index);
    auto four_steps = v0::Constant::create(ov::element::i64, {1}, {4});
    for (int64_t i = 0; i < 4; ++i) {
        auto start = v0::Constant::create(ov::element::i64, {1}, {i});
        auto index = std::make_shared<v8::Slice>(merge_index, start, merge_count, four_steps, row_index);
        auto flat = flat_index(index);
        replace_input(model, "merger.indices." + std::to_string(i), flat);
    }
    if (merge == 4) {
        auto flat_window = flat_index(window_index);
        replace_input(model, "window_indices", flat_window);
        auto count = std::make_shared<v3::ShapeOf>(window_index);
        auto ordered = range(count);
        auto inverse = std::make_shared<v3::ScatterUpdate>(ordered, window_index, ordered, zero);
        auto flat_inverse = flat_index(inverse);
        replace_input(model, "inverse_window_indices", flat_inverse);
        auto four = v0::Constant::create(ov::element::i64, {}, {4});
        auto group = std::make_shared<v1::Divide>(ordered, four);
        auto query_group = std::make_shared<v0::Unsqueeze>(group, col_index);
        auto key_group = std::make_shared<v0::Unsqueeze>(group, row_index);
        auto same_group = std::make_shared<v1::Equal>(query_group, key_group);
        auto visible = v0::Constant::create(ov::element::f32, {}, {0.0f});
        auto hidden = v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<float>::infinity()});
        auto mask = std::make_shared<v1::Select>(same_group, visible, hidden);
        auto batch_heads = v0::Constant::create(ov::element::i64, {2}, {1, 1});
        auto mask_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch_heads, count, count}, 0);
        auto attention_mask = std::make_shared<v1::Reshape>(mask, mask_shape, false);
        replace_input(model, "attention_mask", attention_mask);
        for (int64_t i = 0; i < 4; ++i) {
            auto start = v0::Constant::create(ov::element::i64, {1}, {i});
            auto index = std::make_shared<v8::Slice>(window_index, start, count, four_steps, row_index);
            auto flat = flat_index(index);
            replace_input(model, "vit_merger.indices." + std::to_string(i), flat);
        }
    }
    add_inputs(model,
               {{"pixel_values", pixels},
                {"position_ids", positions},
                {"window_index", window_index},
                {"merge_index", merge_index},
                {"patch_grid", grid}});
    squeeze_output(model);
}

constexpr std::pair<ProjectorLayout, const char*> layout_names[] = {
    {ProjectorLayout::Unsupported, "unsupported"},
    {ProjectorLayout::Image, "image"},
    {ProjectorLayout::PaddedPatches, "padded_patches"},
    {ProjectorLayout::WindowedPatches, "windowed_patches"},
    {ProjectorLayout::SplitMerger, "split_merger"},
    {ProjectorLayout::PackedPatches, "packed_patches"},
};

ProjectorLayout read_projector_layout(const std::shared_ptr<ov::Model>& model) {
    if (!model->has_rt_info({"gguf_mmproj", "vision.genai_layout"})) {
        OPENVINO_ASSERT(model->has_rt_info({"gguf_mmproj", "vision.projector"}),
                        "[GGUF] no GenAI vision layout contract");
        const auto projector = model->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"});
        const auto layout = builtin_projector_layout(projector);
        OPENVINO_ASSERT(layout != ProjectorLayout::Unsupported,
                        "[GGUF] no GenAI vision layout for mmproj projector '",
                        projector,
                        "'");
        return layout;
    }
    const auto name = model->get_rt_info<std::string>({"gguf_mmproj", "vision.genai_layout"});
    for (const auto& [layout, layout_name] : layout_names) {
        if (layout != ProjectorLayout::Unsupported && name == layout_name)
            return layout;
    }
    OPENVINO_THROW("[GGUF] no GenAI vision layout for mmproj contract '", name, "'");
}
}  // namespace

std::string projector_layout_name(ProjectorLayout layout) {
    for (const auto& [candidate, name] : layout_names) {
        if (candidate == layout)
            return name;
    }
    OPENVINO_THROW("[GGUF] invalid GenAI projector layout contract");
}

ProjectorLayout builtin_projector_layout(const std::string& projector_type) {
    if (projector_type == "gemma3" || projector_type == "lfm2")
        return ProjectorLayout::Image;
    if (projector_type == "gemma4v" || projector_type == "gemma4uv")
        return ProjectorLayout::PaddedPatches;
    if (projector_type == "muse-glimmer")
        return ProjectorLayout::WindowedPatches;
    if (projector_type == "qwen3vl_merger")
        return ProjectorLayout::SplitMerger;
    if (projector_type == "minicpmv4_7")
        return ProjectorLayout::PackedPatches;
    return ProjectorLayout::Unsupported;
}

void require_projector_layout(const std::shared_ptr<ov::Model>& model) {
    read_projector_layout(model);
}

ProjectorModels adapt_projector_layout(const std::shared_ptr<ov::Model>& model) {
    switch (read_projector_layout(model)) {
    case ProjectorLayout::Image:
        break;
    case ProjectorLayout::PaddedPatches:
        padded_patch_layout(model, mmproj_int_metadata(model, "vision.patch_size"));
        break;
    case ProjectorLayout::WindowedPatches:
        windowed_patch_layout(model,
                              mmproj_int_metadata(model, "vision.patch_size"),
                              mmproj_int_metadata(model, "vision.window_size"),
                              mmproj_int_metadata(model, "vision.merge"));
        break;
    case ProjectorLayout::SplitMerger:
        return split_merger_layout(model);
    case ProjectorLayout::PackedPatches:
        packed_patch_layout(model,
                            mmproj_int_metadata(model, "vision.patch_size"),
                            mmproj_int_metadata(model, "vision.merge"));
        break;
    case ProjectorLayout::Unsupported:
        OPENVINO_THROW("[GGUF] no GenAI vision layout contract");
    }
    return {{"vision_embeddings", model}};
}

}  // namespace ov::frontend::gguf
