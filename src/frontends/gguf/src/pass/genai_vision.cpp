// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/gguf/genai_vision.hpp"

#include <limits>

#include "openvino/core/graph_util.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf {
namespace {
using namespace ov::op;
using Value = ov::Output<ov::Node>;
using Models = std::map<std::string, std::shared_ptr<ov::Model>>;

Value i64(const std::vector<int64_t>& values) {
    return v0::Constant::create(ov::element::i64, {values.size()}, values);
}
Value scalar(int64_t value) {
    return v0::Constant::create(ov::element::i64, {}, {value});
}
Value add(const Value& a, const Value& b) {
    return std::make_shared<v1::Add>(a, b);
}
Value mul(const Value& a, const Value& b) {
    return std::make_shared<v1::Multiply>(a, b);
}
Value div(const Value& a, const Value& b) {
    return std::make_shared<v1::Divide>(a, b);
}
Value mod(const Value& a, const Value& b) {
    return std::make_shared<v1::FloorMod>(a, b);
}
Value add(const Value& a, int64_t b) {
    return add(a, scalar(b));
}
Value mul(const Value& a, int64_t b) {
    return mul(a, scalar(b));
}
Value div(const Value& a, int64_t b) {
    return div(a, scalar(b));
}
Value mod(const Value& a, int64_t b) {
    return mod(a, scalar(b));
}
Value reshape(const Value& value, const Value& shape) {
    return std::make_shared<v1::Reshape>(value, shape, false);
}
Value as_index(const Value& value) {
    return reshape(value, i64({1, 1, 1, -1}));
}
Value column(const Value& matrix, int64_t index) {
    return std::make_shared<v8::Gather>(matrix, scalar(index), scalar(1));
}
Value range(const Value& count) {
    auto stop = std::make_shared<v0::Squeeze>(count, i64({0}));
    return std::make_shared<v4::Range>(scalar(0), stop, scalar(1), ov::element::i64);
}

std::shared_ptr<v0::Parameter> parameter(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    for (const auto& candidate : model->get_parameters()) {
        if (candidate->get_friendly_name() == name)
            return candidate;
    }
    OPENVINO_THROW("[GGUF] mmproj vision encoder has no input ", name);
}

void replace_input(const std::shared_ptr<ov::Model>& model, const std::string& name, Value value) {
    auto source = parameter(model, name);
    if (value.get_element_type() != source->get_element_type())
        value = std::make_shared<v0::Convert>(value, source->get_element_type());
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

std::shared_ptr<v0::Parameter> new_input(ov::element::Type type, const ov::PartialShape& shape) {
    return std::make_shared<v0::Parameter>(type, shape);
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

int64_t metadata(const std::shared_ptr<ov::Model>& model, const std::string& key) {
    return std::stoll(model->get_rt_info<std::string>({"gguf_mmproj", key}));
}

Value image_from_patches(const Value& patches,
                         const Value& rows,
                         const Value& cols,
                         int64_t patch,
                         bool channels_last) {
    const auto p = i64({patch}), c = i64({3});
    auto grid = channels_last ? ov::OutputVector{rows, cols, p, p, c} : ov::OutputVector{rows, cols, c, p, p};
    Value image = reshape(patches, std::make_shared<v0::Concat>(grid, 0));
    image = std::make_shared<v1::Transpose>(
        image,
        i64(channels_last ? std::vector<int64_t>{4, 0, 2, 1, 3} : std::vector<int64_t>{2, 0, 3, 1, 4}));
    return reshape(image,
                   std::make_shared<v0::Concat>(ov::OutputVector{i64({1, 3}), mul(rows, patch), mul(cols, patch)}, 0));
}

void squeeze_output(const std::shared_ptr<ov::Model>& model) {
    auto result = model->get_results().front();
    auto features = std::make_shared<v0::Squeeze>(result->input_value(0), i64({0}));
    features->output(0).get_tensor().set_names({"last_hidden_state"});
    result->input(0).replace_source_output(features);
}

// Patches fill the leading rows in raster order; padding rows have position -1.
void gemma4_layout(const std::shared_ptr<ov::Model>& model, int64_t patch) {
    auto pixels = new_input(ov::element::f32, {1, -1, 3 * patch * patch});
    auto positions = new_input(ov::element::i64, {1, -1, 2});
    auto flat = std::make_shared<v0::Squeeze>(positions, i64({0}));
    auto x = column(flat, 0), y = column(flat, 1);
    auto count = std::make_shared<v1::ReduceSum>(
        std::make_shared<v0::Convert>(std::make_shared<v1::GreaterEqual>(x, scalar(0)), ov::element::i64),
        i64({0}),
        true);
    auto cols = add(std::make_shared<v1::ReduceMax>(x, i64({0}), true), 1);
    auto rows = div(count, cols);
    const auto valid = [&](const Value& value) {
        return std::make_shared<v8::Slice>(value, i64({0}), count, i64({1}), i64({0}));
    };
    replace_input(model,
                  "pixel_values",
                  image_from_patches(valid(std::make_shared<v0::Squeeze>(pixels, i64({0}))), rows, cols, patch, true));
    replace_input(model, "position_x", as_index(valid(x)));
    replace_input(model, "position_y", as_index(valid(y)));
    add_inputs(model, {{"pixel_values", pixels}, {"image_position_ids", positions}});
}

void muse_layout(const std::shared_ptr<ov::Model>& model, int64_t patch, int64_t window, int64_t merge) {
    auto pixels = new_input(ov::element::f32, {-1, 3 * patch * patch});
    auto grid = new_input(ov::element::i64, {1, 3});
    auto thw = std::make_shared<v0::Squeeze>(grid, i64({0}));
    Value rows = std::make_shared<v8::Gather>(thw, i64({1}), scalar(0));
    Value cols = std::make_shared<v8::Gather>(thw, i64({2}), scalar(0));
    replace_input(model, "pixel_values", image_from_patches(pixels, rows, cols, patch, false));

    // Slots outside the image repeat the window's first patch and are masked out as keys.
    const int64_t slots = window * window;
    auto window_cols = div(add(cols, window - 1), window);
    auto slot = range(mul(mul(div(add(rows, window - 1), window), window_cols), slots));
    auto index_in_window = mod(slot, slots);
    auto window_index = div(slot, slots);
    auto top = mul(div(window_index, window_cols), window);
    auto left = mul(mod(window_index, window_cols), window);
    auto y = add(top, div(index_in_window, window));
    auto x = add(left, mod(index_in_window, window));
    auto inside =
        std::make_shared<v1::LogicalAnd>(std::make_shared<v1::Less>(y, rows), std::make_shared<v1::Less>(x, cols));
    Value index = std::make_shared<v1::Select>(inside, add(mul(y, cols), x), add(mul(top, cols), left));
    replace_input(model, "patch_indices", as_index(index));
    replace_input(model, "position_x", as_index(add(mod(index, cols), 1)));
    replace_input(model, "position_y", as_index(add(div(index, cols), 1)));
    auto key_mask = std::make_shared<v1::Select>(
        inside,
        v0::Constant::create(ov::element::f32, {}, {0.f}),
        v0::Constant::create(ov::element::f32, {}, {-std::numeric_limits<float>::infinity()}));
    replace_input(model, "window_mask", reshape(key_mask, i64({-1, 1, 1, slots})));

    auto raster = range(mul(rows, cols));
    auto r = div(raster, cols), c = mod(raster, cols);
    auto slot_of = add(mul(add(mul(div(r, window), window_cols), div(c, window)), slots),
                       add(mul(mod(r, window), window), mod(c, window)));
    replace_input(model, "output_indices", as_index(slot_of));
    auto block = div(raster, merge * merge), within = mod(raster, merge * merge);
    auto block_cols = div(cols, merge);
    auto merged = add(mul(add(mul(div(block, block_cols), merge), div(within, merge)), cols),
                      add(mul(mod(block, block_cols), merge), mod(within, merge)));
    replace_input(model, "merge_indices", as_index(merged));
    add_inputs(model, {{"pixel_values", pixels}, {"image_grid_thw", grid}});
    squeeze_output(model);
}

std::shared_ptr<ov::Model> single_output_model(const Value& output,
                                               const std::string& name,
                                               const ov::ParameterVector& inputs) {
    auto result = std::make_shared<v0::Result>(output);
    output.get_tensor().set_names({name});
    return std::make_shared<ov::Model>(ov::ResultVector{result}, inputs)->clone();
}

Models qwen_layout(const std::shared_ptr<ov::Model>& model) {
    Models models;
    // HF flattens patches as (channel, frame, y, x).
    auto frame = [&](const std::string& name) {
        return std::make_shared<v0::Unsqueeze>(require_weight(model, name), i64({2}));
    };
    auto kernel =
        std::make_shared<v0::Concat>(ov::OutputVector{frame("v.patch_embd.weight"), frame("v.patch_embd.weight.1")}, 2);
    const auto& kernel_shape = kernel->get_output_partial_shape(0);
    const auto width = kernel_shape[0].get_length();
    const auto patch_dim = ov::shape_size(kernel_shape.get_shape()) / width;
    auto patches = new_input(ov::element::f32, {-1, int64_t(patch_dim)});
    name_output(patches, "hidden_states");
    Value embeddings =
        std::make_shared<v0::MatMul>(patches, reshape(kernel, i64({width, int64_t(patch_dim)})), false, true);
    if (auto bias = weight(model, "v.patch_embd.bias"); bias.get_node())
        embeddings = add(embeddings, bias);
    models["vision_embeddings"] = single_output_model(embeddings, "last_hidden_state", {patches});

    auto table_indices = new_input(ov::element::i64, {4, -1});
    name_output(table_indices, "input");
    auto table =
        std::make_shared<v8::Gather>(require_weight(model, "v.position_embd.weight"), table_indices, scalar(0));
    models["vision_embeddings_pos"] = single_output_model(table, "last_hidden_state", {table_indices});

    auto norm_weight = require_weight(model, "v.blk.0.ln1.weight");
    auto scale = norm_weight.get_target_inputs().begin()->get_node();
    auto normalized = scale->input_value(0) == norm_weight ? scale->input_value(1) : scale->input_value(0);
    auto hidden_states = new_input(ov::element::f32, {-1, width});
    normalized.get_node()->input_value(0).replace(std::make_shared<v0::Unsqueeze>(hidden_states, i64({0, 1})));
    model->remove_parameter(parameter(model, "pixel_values"));
    model->remove_parameter(parameter(model, "patch_indices"));

    // The first column of each rotary half has frequency 1, so it holds the row or column.
    const auto head =
        metadata(model, "clip.vision.embedding_length") / metadata(model, "clip.vision.attention.head_count");
    auto rotary = new_input(ov::element::f32, {-1, head / 2});
    auto row = column(rotary, 0), col = column(rotary, head / 4);
    auto positions = std::make_shared<v5::Round>(std::make_shared<v0::Concat>(ov::OutputVector{row, col, row, col}, 0),
                                                 v5::Round::RoundMode::HALF_TO_EVEN);
    replace_input(model, "position_ids", as_index(positions));

    auto mask = new_input(ov::element::f32, {1, -1, -1});
    auto mask4 = std::make_shared<v0::Unsqueeze>(mask, i64({0}));
    for (const auto& node : model->get_ordered_ops()) {
        if (auto sdpa = ov::as_type_ptr<v13::ScaledDotProductAttention>(node)) {
            OPENVINO_ASSERT(sdpa->get_input_size() == 3, "[GGUF] Qwen vision attention already has a mask");
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
        model->add_results({std::make_shared<v0::Result>(deepstack)});
    }
    add_inputs(model, {{"hidden_states", hidden_states}, {"attention_mask", mask}, {"rotary_pos_emb", rotary}});
    models["vision_embeddings_merger"] = model;
    return models;
}
}  // namespace

std::map<std::string, std::shared_ptr<ov::Model>> genai_vision_models(const std::shared_ptr<ov::Model>& mmproj) {
    auto vision = mmproj->clone();
    pass::AdaptMmprojToGenAI(pass::AdaptMmprojToGenAI::Modality::VISION).run_on_model(vision);
    const auto projector = vision->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"});
    if (projector == "qwen3vl_merger")
        return qwen_layout(vision);
    if (projector == "gemma4v" || projector == "gemma4uv") {
        gemma4_layout(vision, metadata(vision, "vision.patch_size"));
    } else if (projector == "muse-glimmer") {
        muse_layout(vision,
                    metadata(vision, "vision.patch_size"),
                    metadata(vision, "vision.window_size"),
                    metadata(vision, "vision.merge"));
    } else {
        OPENVINO_ASSERT(projector == "gemma3", "[GGUF] no GenAI vision layout for mmproj projector '", projector, "'");
    }
    return {{"vision_embeddings", vision}};
}

}  // namespace ov::frontend::gguf
