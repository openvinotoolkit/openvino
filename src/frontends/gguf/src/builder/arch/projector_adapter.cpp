// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "projector_adapter.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <utility>

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
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
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pass/prune_orphaned_parameters.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {
namespace {
using namespace ov::op;
using Value = ov::Output<ov::Node>;

std::shared_ptr<v0::Constant> ints(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, {values.size()}, values);
}

std::shared_ptr<v0::Constant> scalar(int64_t value) {
    return v0::Constant::create(ov::element::i64, {}, {value});
}

Value add(const Value& a, const Value& b) {
    return std::make_shared<v1::Add>(a, b);
}

Value multiply(const Value& a, const Value& b) {
    return std::make_shared<v1::Multiply>(a, b);
}

Value divide(const Value& a, const Value& b) {
    return std::make_shared<v1::Divide>(a, b);
}

Value modulo(const Value& a, const Value& b) {
    return std::make_shared<v1::FloorMod>(a, b);
}

Value range(const Value& count) {
    auto stop = std::make_shared<v0::Squeeze>(count, ints({0}));
    return std::make_shared<v4::Range>(scalar(0), stop, scalar(1), ov::element::i64);
}

std::shared_ptr<v0::Parameter> parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto candidate = find_parameter(model, name);
    OPENVINO_ASSERT(candidate, "[GGUF] mmproj vision encoder has no input ", name);
    return candidate;
}

Value flat_index(const Value& value) {
    return std::make_shared<v1::Reshape>(value, ints({1, 1, 1, -1}), false);
}

Value additive_mask(const Value& visible) {
    auto unmasked = v0::Constant::create(ov::element::f32, {}, {0.f});
    auto masked = v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<float>::infinity()});
    return std::make_shared<v1::Select>(visible, unmasked, masked);
}

void require_square_fits(int64_t value, int64_t factor, const char* key) {
    OPENVINO_ASSERT(value > 0 && value <= std::numeric_limits<int64_t>::max() / factor / value,
                    "[GGUF] out-of-range mmproj metadata ",
                    key);
}

void replace_input(const std::shared_ptr<ov::Model>& model, const std::string& name, Value value) {
    auto source = parameter(model, name);
    if (value.get_element_type() != source->get_element_type())
        value = std::make_shared<v0::Convert>(value, source->get_element_type());
    source->output(0).replace(value);
    value.get_tensor().set_names({});
    model->remove_parameter(source);
}

// Replaces "<prefix>0".."<prefix>3" with the four interleaved quarters of `source`.
void replace_strided_inputs(const std::shared_ptr<ov::Model>& model,
                            const Value& source,
                            const Value& count,
                            const std::string& prefix) {
    for (int64_t i = 0; i < 4; ++i) {
        auto quarter = std::make_shared<v8::Slice>(source, ints({i}), count, ints({4}), ints({0}));
        replace_input(model, prefix + std::to_string(i), flat_index(quarter));
    }
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
    for (const auto& node : model->get_ops()) {
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

Value image_shape(const Value& rows, const Value& cols, int64_t patch) {
    auto patch_size = scalar(patch);
    return std::make_shared<v0::Concat>(
        ov::OutputVector{ints({1, 3}), multiply(rows, patch_size), multiply(cols, patch_size)},
        0);
}

Value image_from_patches(const Value& patches,
                         const Value& rows,
                         const Value& cols,
                         int64_t patch,
                         bool channels_last) {
    auto p = ints({patch});
    auto c = ints({3});
    auto grid = channels_last ? ov::OutputVector{rows, cols, p, p, c} : ov::OutputVector{rows, cols, c, p, p};
    auto grid_shape = std::make_shared<v0::Concat>(grid, 0);
    auto patch_grid = std::make_shared<v1::Reshape>(patches, grid_shape, false);
    auto order = channels_last ? ints({4, 0, 2, 1, 3}) : ints({2, 0, 3, 1, 4});
    auto image_grid = std::make_shared<v1::Transpose>(patch_grid, order);
    return std::make_shared<v1::Reshape>(image_grid, image_shape(rows, cols, patch), false);
}

void squeeze_output(const std::shared_ptr<ov::Model>& model) {
    auto result = model->get_results().front();
    auto features = std::make_shared<v0::Squeeze>(result->input_value(0), ints({0}));
    features->output(0).get_tensor().set_names({"last_hidden_state"});
    result->input(0).replace_source_output(features);
}

// Patches fill the leading rows in raster order; padding rows have position -1.
void padded_patch_layout(const std::shared_ptr<ov::Model>& model, int64_t patch) {
    require_square_fits(patch, 3, "vision.patch_size");
    auto pixels = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 3 * patch * patch});
    auto positions = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1, 2});
    auto zero_axis = ints({0});
    auto zero = scalar(0);
    auto one = scalar(1);
    auto flat = std::make_shared<v0::Squeeze>(positions, zero_axis);
    auto x = std::make_shared<v8::Gather>(flat, zero, one);
    auto y = std::make_shared<v8::Gather>(flat, one, one);
    auto nonnegative = std::make_shared<v1::GreaterEqual>(x, zero);
    auto valid_rows = std::make_shared<v0::Convert>(nonnegative, ov::element::i64);
    auto count = std::make_shared<v1::ReduceSum>(valid_rows, zero_axis, true);
    auto cols = add(std::make_shared<v1::ReduceMax>(x, zero_axis, true), one);
    auto rows = divide(count, cols);
    auto leading = [&](const Value& value) -> Value {
        return std::make_shared<v8::Slice>(value, zero_axis, count, ints({1}), zero_axis);
    };
    auto flat_pixels = std::make_shared<v0::Squeeze>(pixels, zero_axis);
    replace_input(model, "pixel_values", image_from_patches(leading(flat_pixels), rows, cols, patch, true));
    replace_input(model, "position_x", flat_index(leading(x)));
    replace_input(model, "position_y", flat_index(leading(y)));
    add_inputs(model, {{"pixel_values", pixels}, {"image_position_ids", positions}});
}

void windowed_patch_layout(const std::shared_ptr<ov::Model>& model, int64_t patch, int64_t window, int64_t merge) {
    require_square_fits(patch, 3, "vision.patch_size");
    require_square_fits(window, 1, "vision.window_size");
    require_square_fits(merge, 1, "vision.merge");
    auto pixels = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 3 * patch * patch});
    auto grid = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3});
    auto thw = std::make_shared<v0::Squeeze>(grid, ints({0}));
    auto rows = gather_dims(thw, {1});
    auto cols = gather_dims(thw, {2});
    replace_input(model, "pixel_values", image_from_patches(pixels, rows, cols, patch, false));

    // Slots outside the image repeat the window's first patch and are masked out as keys.
    const int64_t slots = window * window;
    auto window_size = scalar(window);
    auto window_padding = scalar(window - 1);
    auto window_slots = scalar(slots);
    auto window_cols = divide(add(cols, window_padding), window_size);
    auto window_rows = divide(add(rows, window_padding), window_size);
    auto slot = range(multiply(multiply(window_rows, window_cols), window_slots));
    auto index_in_window = modulo(slot, window_slots);
    auto window_index = divide(slot, window_slots);
    auto top = multiply(divide(window_index, window_cols), window_size);
    auto left = multiply(modulo(window_index, window_cols), window_size);
    auto y = add(top, divide(index_in_window, window_size));
    auto x = add(left, modulo(index_in_window, window_size));
    auto inside =
        std::make_shared<v1::LogicalAnd>(std::make_shared<v1::Less>(y, rows), std::make_shared<v1::Less>(x, cols));
    auto index = std::make_shared<v1::Select>(inside, add(multiply(y, cols), x), add(multiply(top, cols), left));
    auto one = scalar(1);
    replace_input(model, "patch_indices", flat_index(index));
    replace_input(model, "position_x", flat_index(add(modulo(index, cols), one)));
    replace_input(model, "position_y", flat_index(add(divide(index, cols), one)));
    auto window_mask = std::make_shared<v1::Reshape>(additive_mask(inside), ints({-1, 1, 1, slots}), false);
    replace_input(model, "window_mask", window_mask);

    auto raster = range(multiply(rows, cols));
    auto r = divide(raster, cols);
    auto c = modulo(raster, cols);
    auto raster_window = add(multiply(divide(r, window_size), window_cols), divide(c, window_size));
    auto raster_in_window = add(multiply(modulo(r, window_size), window_size), modulo(c, window_size));
    replace_input(model, "output_indices", flat_index(add(multiply(raster_window, window_slots), raster_in_window)));

    auto merge_size = scalar(merge);
    auto merge_slots = scalar(merge * merge);
    auto block = divide(raster, merge_slots);
    auto within = modulo(raster, merge_slots);
    auto block_cols = divide(cols, merge_size);
    auto merged_row = add(multiply(divide(block, block_cols), merge_size), divide(within, merge_size));
    auto merged_col = add(multiply(modulo(block, block_cols), merge_size), modulo(within, merge_size));
    replace_input(model, "merge_indices", flat_index(add(multiply(merged_row, cols), merged_col)));
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

ProjectorModels split_merger_layout(const std::shared_ptr<ov::Model>& model) {
    ProjectorModels models;
    auto live_before = std::make_shared<std::unordered_set<const ov::Node*>>();
    pass::SnapshotLiveParameters(live_before).run_on_model(model);
    // HF flattens patches as (channel, frame, y, x).
    auto frame = [](const Value& kernel) {
        return std::make_shared<v0::Unsqueeze>(kernel, ints({2}));
    };
    auto first_frame = frame(require_weight(model, "v.patch_embd.weight"));
    auto second_frame = frame(require_weight(model, "v.patch_embd.weight.1"));
    auto kernel = std::make_shared<v0::Concat>(ov::OutputVector{first_frame, second_frame}, 2);
    const auto& kernel_shape = kernel->get_output_partial_shape(0);
    const auto width = kernel_shape[0].get_length();
    const auto patch_dim = static_cast<int64_t>(ov::shape_size(kernel_shape.get_shape()) / width);
    auto patches = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, patch_dim});
    name_output(patches, "hidden_states");
    auto kernel_2d = std::make_shared<v1::Reshape>(kernel, ints({width, patch_dim}), false);
    Value embeddings = std::make_shared<v0::MatMul>(patches, kernel_2d, false, true);
    if (auto bias = weight(model, "v.patch_embd.bias"); bias.get_node())
        embeddings = add(embeddings, bias);
    models["vision_embeddings"] = single_output_model(embeddings, "last_hidden_state", {patches});

    auto table_indices = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{4, -1});
    name_output(table_indices, "input");
    auto position_weight = require_weight(model, "v.position_embd.weight");
    auto zero = scalar(0);
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
    auto hidden_states_4d = std::make_shared<v0::Unsqueeze>(hidden_states, ints({0, 1}));
    // Replace the activation shared by LayerNorm and its residual path.
    normalized->input_value(0).replace(hidden_states_4d);
    pass::PruneParametersOrphanedSince(live_before).run_on_model(model);

    // The first column of each rotary half has frequency 1, so it holds the row or column.
    const auto width_metadata = mmproj_int_metadata(model, "clip.vision.embedding_length");
    const auto heads = mmproj_int_metadata(model, "clip.vision.attention.head_count");
    OPENVINO_ASSERT(width_metadata == width && width_metadata % heads == 0 && width_metadata / heads % 4 == 0,
                    "[GGUF] incompatible mmproj metadata clip.vision.embedding_length and "
                    "clip.vision.attention.head_count");
    const auto head = width_metadata / heads;
    auto rotary = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, head / 2});
    auto one = scalar(1);
    auto row = std::make_shared<v8::Gather>(rotary, zero, one);
    auto col = std::make_shared<v8::Gather>(rotary, scalar(head / 4), one);
    auto coordinates = std::make_shared<v0::Concat>(ov::OutputVector{row, col, row, col}, 0);
    auto positions = std::make_shared<v5::Round>(coordinates, v5::Round::RoundMode::HALF_TO_EVEN);
    replace_input(model, "position_ids", flat_index(positions));

    auto mask = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, -1});
    auto mask4 = std::make_shared<v0::Unsqueeze>(mask, ints({0}));
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
    ov::OutputVector levels;
    for (size_t i = 1; i < results.size(); ++i) {
        const auto& names = results[i]->output(0).get_names();
        if (std::any_of(names.begin(), names.end(), [](const std::string& name) {
                return name.rfind("deepstack_features.", 0) == 0;
            })) {
            levels.push_back(results[i]->input_value(0));
            model->remove_result(results[i]);
        }
    }
    if (!levels.empty()) {
        auto deepstack = std::make_shared<v0::Concat>(levels, 0);
        deepstack->output(0).get_tensor().set_names({"deepstack_feature_lists"});
        model->add_results({std::make_shared<v0::Result>(deepstack)});
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
    auto rows = gather_dims(grid, {0});
    auto cols = gather_dims(grid, {1});
    auto strip = std::make_shared<v1::Reshape>(pixels, ints({1, 3, patch, -1, patch}), false);
    auto reordered = std::make_shared<v1::Transpose>(strip, ints({0, 3, 1, 2, 4}));
    auto packed = std::make_shared<v1::Reshape>(reordered, ints({-1, 3 * patch * patch}), false);
    replace_input(model, "pixel_values", image_from_patches(packed, rows, cols, patch, false));
    replace_input(model, "position_ids", flat_index(positions));
    replace_strided_inputs(model, merge_index, std::make_shared<v3::ShapeOf>(merge_index), "merger.indices.");
    if (merge == 4) {
        replace_input(model, "window_indices", flat_index(window_index));
        auto count = std::make_shared<v3::ShapeOf>(window_index);
        auto ordered = range(count);
        auto inverse = std::make_shared<v3::ScatterUpdate>(ordered, window_index, ordered, scalar(0));
        replace_input(model, "inverse_window_indices", flat_index(inverse));
        auto group = divide(ordered, scalar(4));
        auto query_group = std::make_shared<v0::Unsqueeze>(group, ints({1}));
        auto key_group = std::make_shared<v0::Unsqueeze>(group, ints({0}));
        auto same_group = std::make_shared<v1::Equal>(query_group, key_group);
        auto mask_shape = std::make_shared<v0::Concat>(ov::OutputVector{ints({1, 1}), count, count}, 0);
        auto attention_mask = std::make_shared<v1::Reshape>(additive_mask(same_group), mask_shape, false);
        replace_input(model, "attention_mask", attention_mask);
        replace_strided_inputs(model, window_index, count, "vit_merger.indices.");
    }
    add_inputs(model,
               {{"pixel_values", pixels},
                {"position_ids", positions},
                {"window_index", window_index},
                {"merge_index", merge_index},
                {"patch_grid", grid}});
    squeeze_output(model);
}

void temporal_packed_patch_layout(const std::shared_ptr<ov::Model>& model, int64_t patch, int64_t temporal) {
    OPENVINO_ASSERT(temporal == 1 || temporal == 2,
                    "[GGUF] temporal packed vision requires one or two temporal patches");
    require_square_fits(patch, 3 * temporal, "vision.patch_size");
    // The body is a clone so the supplied model stays untouched until the iterator is complete.
    auto body = model->clone();
    auto first_pixels = find_parameter(body, "pixel_values");
    auto second_pixels = find_parameter(body, "pixel_values_1");
    auto spatial_positions = find_parameter(body, "positions");
    OPENVINO_ASSERT(bool(second_pixels) == (temporal == 2),
                    "[GGUF] temporal packed vision input count must match vision.temporal_patch_size");
    OPENVINO_ASSERT(first_pixels && spatial_positions && body->get_parameters().size() == (second_pixels ? 3u : 2u) &&
                        body->get_results().size() == 1 && body->get_sinks().empty() && body->get_variables().empty(),
                    "[GGUF] temporal packed vision requires pixel inputs, positions and one embedding output");
    OPENVINO_ASSERT(spatial_positions->get_element_type() == ov::element::i32,
                    "[GGUF] temporal packed vision requires i32 spatial positions");

    const auto pixel_dim = 3 * temporal * patch * patch;
    auto pixels = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, -1, pixel_dim});
    auto shape = std::make_shared<v3::ShapeOf>(pixels);
    auto rows = gather_dims(shape, {1});
    auto cols = gather_dims(shape, {2});
    auto packed_shape =
        std::make_shared<v0::Concat>(ov::OutputVector{ints({1}), rows, cols, ints({3, temporal, patch, patch})}, 0);
    auto packed = std::make_shared<v1::Reshape>(pixels, packed_shape, false);
    auto frame_shape = image_shape(rows, cols, patch);
    auto replace_frame = [&](const std::shared_ptr<v0::Parameter>& input, int64_t index) {
        const auto& input_shape = input->get_partial_shape();
        OPENVINO_ASSERT(input->get_element_type() == ov::element::f32 && input_shape.rank() == 4 &&
                            input_shape[0] == 1 && input_shape[1] == 3,
                        "[GGUF] temporal packed vision requires singleton RGB NCHW pixel inputs");
        auto frame = std::make_shared<v8::Gather>(packed, scalar(index), scalar(4));
        auto image = std::make_shared<v1::Transpose>(frame, ints({0, 3, 1, 4, 2, 5}));
        replace_input(body, input->get_friendly_name(), std::make_shared<v1::Reshape>(image, frame_shape, false));
    };
    replace_frame(first_pixels, 0);
    if (second_pixels)
        replace_frame(second_pixels, 1);

    auto count = multiply(rows, cols);
    auto indices = range(count);
    auto zeros = std::make_shared<v3::Broadcast>(ints({0}), count);
    auto raster =
        std::make_shared<v0::Concat>(ov::OutputVector{divide(indices, cols), modulo(indices, cols), zeros, zeros}, 0);
    replace_input(body, "positions", flat_index(std::make_shared<v0::Convert>(raster, ov::element::i32)));
    add_inputs(body, {{"pixel_values", pixels}});
    const auto features = body->get_results().front()->input_value(0);
    const auto output_shape = features.get_partial_shape();
    OPENVINO_ASSERT(features.get_element_type() == ov::element::f32 && output_shape.rank() == 3 && output_shape[0] == 1,
                    "[GGUF] temporal packed vision requires f32 [1, tokens, width] embeddings");

    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1, pixel_dim});
    name_output(input, "pixel_values");
    auto iterator = std::make_shared<v0::TensorIterator>();
    iterator->set_body(body);
    iterator->set_sliced_input(pixels, input, 0, 1, 1, -1, 0);
    auto output = iterator->get_concatenated_slices(features, 0, 1, 1, -1, 0);
    output.set_names({"last_hidden_state"});
    iterator->validate_and_infer_types();
    for (const auto& result : ov::ResultVector(model->get_results()))
        model->remove_result(result);
    for (const auto& parameter : ov::ParameterVector(model->get_parameters()))
        model->remove_parameter(parameter);
    model->add_parameters({input});
    model->add_results({std::make_shared<v0::Result>(output)});
    model->validate_nodes_and_infer_types();
}

constexpr std::pair<ProjectorLayout, const char*> layout_names[] = {
    {ProjectorLayout::Unsupported, "unsupported"},
    {ProjectorLayout::Image, "image"},
    {ProjectorLayout::PaddedPatches, "padded_patches"},
    {ProjectorLayout::WindowedPatches, "windowed_patches"},
    {ProjectorLayout::SplitMerger, "split_merger"},
    {ProjectorLayout::PackedPatches, "packed_patches"},
    {ProjectorLayout::TemporalPackedPatches, "temporal_packed_patches"},
};
}  // namespace

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

std::string projector_layout_name(ProjectorLayout layout) {
    for (const auto& [candidate, name] : layout_names) {
        if (candidate == layout)
            return name;
    }
    OPENVINO_THROW("[GGUF] invalid GenAI projector layout contract");
}

std::vector<std::string> projector_embedding_outputs(const std::shared_ptr<ov::Model>& model,
                                                     const std::string& modality) {
    const auto key = modality + ".embedding_output_count";
    std::vector<std::string> outputs;
    if (model->has_rt_info({"gguf_mmproj", key})) {
        const auto count = mmproj_int_metadata(model, key);
        OPENVINO_ASSERT(count > 0, "[GGUF] projector embedding output count must be positive");
        std::unordered_set<std::string> names;
        for (int64_t i = 0; i < count; ++i) {
            const auto name_key = modality + ".embedding_output." + std::to_string(i);
            OPENVINO_ASSERT(model->has_rt_info({"gguf_mmproj", name_key}),
                            "[GGUF] missing projector embedding output '",
                            name_key,
                            "'");
            auto name = model->get_rt_info<std::string>({"gguf_mmproj", name_key});
            OPENVINO_ASSERT(!name.empty() && names.insert(name).second,
                            "[GGUF] projector embedding output names must be nonempty and unique");
            outputs.push_back(std::move(name));
        }
    } else {
        const auto auxiliary = modality == "vision" && model->has_rt_info({"gguf_mmproj", "vision.auxiliary_count"})
                                   ? mmproj_int_metadata(model, "vision.auxiliary_count", 0)
                                   : 0;
        OPENVINO_ASSERT(auxiliary >= 0, "[GGUF] projector auxiliary count must be nonnegative");
        outputs.push_back(modality == "vision" ? "image_features" : "audio_features");
        for (int64_t i = 0; i < auxiliary; ++i)
            outputs.push_back("deepstack_features." + std::to_string(i));
    }
    return outputs;
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
    case ProjectorLayout::TemporalPackedPatches:
        temporal_packed_patch_layout(model,
                                     mmproj_int_metadata(model, "vision.patch_size"),
                                     mmproj_int_metadata(model, "vision.temporal_patch_size"));
        break;
    case ProjectorLayout::Unsupported:
        OPENVINO_THROW("[GGUF] no GenAI vision layout contract");
    }
    return {{"vision_embeddings", model}};
}

}  // namespace ov::frontend::gguf
