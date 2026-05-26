// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

#include "model_builder_internal.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

namespace {
void annotate_constants_with_weightless_cache(const std::shared_ptr<ov::Model>& model) {
    std::size_t offset = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::op::util::is_constant(node)) {
            continue;
        }
        const auto& c = std::static_pointer_cast<ov::op::v0::Constant>(node);
        c->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
            ov::WeightlessCacheAttribute(c->get_byte_size(), offset, c->get_element_type());
        offset += c->get_byte_size();
    }
}
}  // namespace

ov::Output<ov::Node> make_linear(const ov::Output<ov::Node>& input,
                                 size_t in_features,
                                 size_t out_features,
                                 const std::string& name,
                                 ov::element::Type precision,
                                 const WeightFn& weight_fn,
                                 const WeightFn& bias_fn) {
    auto weight_output = weight_fn(name + ".weight", ov::Shape{out_features, in_features}, precision);

    auto matmul = std::make_shared<ov::opset11::MatMul>(input, weight_output, false, true);
    matmul->set_friendly_name(name);

    if (bias_fn) {
        auto bias = bias_fn(name + ".bias", ov::Shape{out_features}, precision);
        auto add = std::make_shared<ov::opset11::Add>(matmul, bias);
        add->set_friendly_name(name + "_bias_add");
        return add->output(0);
    }

    return matmul->output(0);
}


ov::Output<ov::Node> make_embedding(const ov::Output<ov::Node>& input_ids,
                                    size_t vocab_size,
                                    size_t hidden_size,
                                    const std::string& name,
                                    ov::element::Type precision) {
    // Per-element PRNG so each token gets a distinct embedding vector.
    uint32_t state = seed_from_name(name);
    size_t total = vocab_size * hidden_size;
    std::vector<float> data(total);
    for (size_t i = 0; i < total; ++i) {
        uint32_t r = xorshift32(state);
        data[i] = static_cast<float>(r % 10000u) / 10000.0f - 0.5f;
    }
    auto weight = ov::opset11::Constant::create(precision, ov::Shape{vocab_size, hidden_size}, data);
    weight->set_friendly_name(name + ".weight");

    auto axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});

    auto gather = std::make_shared<ov::opset11::Gather>(weight, input_ids, axis, 0);
    gather->set_friendly_name(name);

    return gather->output(0);
}

ov::Output<ov::Node> make_lm_head(const ov::Output<ov::Node>& hidden_states,
                                  size_t hidden_size,
                                  size_t vocab_size,
                                  const std::string& name,
                                  ov::element::Type precision,
                                  const WeightFn& weight_fn) {
    return make_linear(hidden_states, hidden_size, vocab_size, name, precision, weight_fn);
}

ov::Output<ov::Node> make_conv1d(const ov::Output<ov::Node>& input,
                                 size_t in_channels,
                                 size_t out_channels,
                                 size_t kernel_size,
                                 size_t stride,
                                 size_t padding,
                                 const std::string& name,
                                 ov::element::Type precision) {
    float w_val = fill_value_from_name(name + ".weight");
    float b_val = fill_value_from_name(name + ".bias") * 0.01f;
    auto weight = ov::opset11::Constant::create(precision,
                                                ov::Shape{out_channels, in_channels, kernel_size},
                                                std::vector<float>(out_channels * in_channels * kernel_size, w_val));
    weight->set_friendly_name(name + ".weight");

    auto conv = std::make_shared<ov::op::v1::Convolution>(input,
                                                          weight,
                                                          ov::Strides{stride},
                                                          ov::CoordinateDiff{static_cast<std::ptrdiff_t>(padding)},
                                                          ov::CoordinateDiff{static_cast<std::ptrdiff_t>(padding)},
                                                          ov::Strides{1});
    conv->set_friendly_name(name);

    auto bias = ov::opset11::Constant::create(precision,
                                              ov::Shape{1, out_channels, 1},
                                              std::vector<float>(out_channels, b_val));
    bias->set_friendly_name(name + ".bias");

    auto add = std::make_shared<ov::opset11::Add>(conv, bias);
    add->set_friendly_name(name + "_bias_add");

    return add->output(0);
}


ov::Output<ov::Node> make_transformer_layers(const ov::Output<ov::Node>& initial,
                                             size_t num_layers,
                                             const std::string& prefix_base,
                                             const LayerFn& layer_fn) {
    ov::Output<ov::Node> current = initial;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        std::string prefix = prefix_base + std::to_string(layer) + ".";
        current = layer_fn(current, prefix, layer);
    }
    return current;
}



std::shared_ptr<ov::Model> ModelBuilder::get_model_with_one_op() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset11::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_without_repeated_blocks() {
    clear();
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
    m_nodes.push_back(input);
    set_name(input);

    std::shared_ptr<ov::Node> res = get_block(input);

    auto result = std::make_shared<ov::op::v0::Result>(res);
    m_nodes.push_back(result);
    set_name(result);

    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks(std::size_t repetitions) {
    return get_model_with_repeated_blocks_and_results(repetitions, {});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks() {
    return get_model_with_repeated_blocks(10);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_with_weightless_cache(std::size_t repetitions) {
    auto model = get_model_with_repeated_blocks(repetitions);
    annotate_constants_with_weightless_cache(model);
    return model;
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_with_weightless_cache() {
    return get_model_with_repeated_blocks_with_weightless_cache(10);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_and_results(
    std::size_t repetitions,
    const std::vector<std::size_t>& block_indices) {
    clear();
    // Generate head
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
    m_nodes.push_back(input);
    set_name(input);

    std::vector<std::shared_ptr<ov::Node>> head(7, nullptr);
    head[0] = std::make_shared<ov::op::v1::Add>(input, input);
    head[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{2});
    head[2] = std::make_shared<ov::op::v1::Divide>(head[0], head[1], true);
    head[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
    head[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int>{1, 1, 40});
    head[5] = std::make_shared<ov::op::v1::Reshape>(head[2], head[3], false);
    head[6] = std::make_shared<ov::op::v1::Reshape>(head[5], head[4], false);

    for (const auto& h : head) {
        m_nodes.push_back(h);
        set_name(h);
    }

    // Generate repeated blocks
    std::shared_ptr<ov::Node> output = get_block(head[6]);
    std::vector<std::shared_ptr<ov::Node>> block_outputs;
    block_outputs.push_back(output);

    for (size_t i = 0; i < repetitions - 1; ++i) {
        output = get_block(output);
        block_outputs.push_back(output);
    }

    // Generate tail
    std::vector<std::shared_ptr<ov::Node>> tail(6, nullptr);
    tail[0] = std::make_shared<ov::op::v0::Concat>(block_outputs, -1);
    tail[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                     ov::Shape{3},
                                                     std::vector<int>{1, 40, int(repetitions)});
    tail[2] = std::make_shared<ov::op::v1::Reshape>(tail[0], tail[1], false);
    tail[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1});
    tail[4] = std::make_shared<ov::op::v1::Multiply>(tail[2], tail[3]);
    tail[5] = std::make_shared<ov::op::v1::Add>(tail[4], tail[4]);

    for (const auto& t : tail) {
        m_nodes.push_back(t);
        set_name(t);
    }

    // Create Results
    ov::OutputVector outputs;

    // Add Results for specified blocks
    for (size_t idx : block_indices) {
        if (idx < block_outputs.size()) {
            auto result = std::make_shared<ov::op::v0::Result>(block_outputs[idx]);
            m_nodes.push_back(result);
            set_name(result);
            outputs.push_back(result->output(0));
        }
    }

    // Always add final tail Result
    auto final_result = std::make_shared<ov::op::v0::Result>(tail[5]);
    m_nodes.push_back(final_result);
    set_name(final_result);
    outputs.push_back(final_result->output(0));

    return std::make_shared<ov::Model>(outputs);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_and_parameters(
    std::size_t repetitions,
    const std::vector<std::size_t>& block_indices) {
    clear();
    if (repetitions == 0) {
        repetitions = 1;
    }

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    m_nodes.push_back(input);

    std::vector<std::size_t> sorted_indices = block_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    sorted_indices.erase(std::unique(sorted_indices.begin(), sorted_indices.end()), sorted_indices.end());

    std::unordered_map<std::size_t, std::shared_ptr<ov::opset11::Parameter>> block_params;
    for (std::size_t idx : sorted_indices) {
        if (idx >= repetitions) {
            continue;
        }

        auto param = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
        m_nodes.push_back(param);
        block_params.emplace(idx, param);
    }

    auto scale_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto bias_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 0.5f));
    auto head_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 1.f));
    m_nodes.push_back(scale_const);
    m_nodes.push_back(bias_const);
    m_nodes.push_back(head_const);

    auto head_add = std::make_shared<ov::opset11::Add>(input, head_const);
    auto head_relu = std::make_shared<ov::opset11::Relu>(head_add);
    m_nodes.push_back(head_add);
    m_nodes.push_back(head_relu);

    ov::Output<ov::Node> current = head_relu;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto it = block_params.find(i);
        ov::Output<ov::Node> rhs = (it != block_params.end()) ? it->second : current;

        auto add = std::make_shared<ov::opset11::Add>(current, rhs);
        m_nodes.push_back(add);

        auto mul = std::make_shared<ov::opset11::Multiply>(add, scale_const);
        m_nodes.push_back(mul);

        auto relu = std::make_shared<ov::opset11::Relu>(mul);
        m_nodes.push_back(relu);

        auto add_bias = std::make_shared<ov::opset11::Add>(relu, bias_const);
        m_nodes.push_back(add_bias);

        current = add_bias;
    }

    auto tail_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 2.f));
    m_nodes.push_back(tail_const);

    auto tail_mul = std::make_shared<ov::opset11::Multiply>(current, tail_const);
    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_const);
    m_nodes.push_back(tail_mul);
    m_nodes.push_back(tail_add);

    auto result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(result);

    return std::make_shared<ov::Model>(ov::OutputVector{result->output(0)});
}

// Builds a model with N identical repeated blocks (using get_block(), same structure as
// get_model_with_repeated_blocks_and_results) where "head" blocks additionally expose
// their output via a MatMul to a separate Parameter-weighted projection group, mimicking
// Gemma4's KV-sharing pattern:
//   - Non-head blocks: block_output → next_block (only internal consumer)
//   - Head blocks:     block_output → next_block  (internal)
// Builds a model with N identical repeated blocks where "head" blocks additionally
// expose their interior Relu via a cross-group MatMul, reproducing the Gemma4
// KV-sharing asymmetry pattern.
//
// Block structure: Add(bias) → Relu(interior/TAP) → Multiply(scale) → Relu(boundary)
//
// Both Relu nodes have identical metadescs (same op type, same f32{1,1,8} I/O shape).
// When a "head" block's interior Relu gains an external MatMul consumer, it becomes
// an additional output_layer — but its metadesc ("Relu…") is already present in
// output_ometa via the boundary Relu.  Therefore ALL blocks retain the same
// MetaInterconnectIO and remain in one repeated-block family, allowing
// isRegularCrossGroupConsumerCase to observe the per-bank inconsistency:
//   - non-head: interior Relu bank → has_external=false (only internal Multiply consumer)
//   - head:     interior Relu bank → has_external=true  (Multiply + external MatMul)
// The mask mismatch causes isRegularCrossGroupConsumerCase to return false →
// irregular_io=true, disabling F16IC for this model.
std::shared_ptr<ov::Model> ModelBuilder::get_model_with_kv_sharing_repeated_blocks(
    std::size_t repetitions,
    const std::vector<std::size_t>& head_block_indices) {
    clear();
    if (repetitions == 0)
        repetitions = 1;

    const std::unordered_set<std::size_t> head_set(head_block_indices.begin(), head_block_indices.end());

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    auto kv_weight = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{8, 4});
    m_nodes.push_back(input);
    m_nodes.push_back(kv_weight);
    set_name(input);
    set_name(kv_weight);

    // Shared constants — same pointer → same Constant node used by all blocks,
    // ensuring identical metadescs for the Add and Multiply ops in every block.
    auto bias_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 0.1f));
    auto scale_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{0.5f});
    m_nodes.push_back(bias_const);
    m_nodes.push_back(scale_const);
    set_name(bias_const);
    set_name(scale_const);

    // Non-repeated prefix — distinct structure prevents it from joining the repetitions.
    auto head_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 1.f));
    auto head_add = std::make_shared<ov::opset11::Add>(input, head_const);
    auto head_relu = std::make_shared<ov::opset11::Relu>(head_add);
    m_nodes.push_back(head_const);
    m_nodes.push_back(head_add);
    m_nodes.push_back(head_relu);
    set_name(head_const);
    set_name(head_add);
    set_name(head_relu);

    ov::Output<ov::Node> current = head_relu;
    ov::OutputVector kv_outputs;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto add_i = std::make_shared<ov::opset11::Add>(current, bias_const);
        auto relu_interior_i = std::make_shared<ov::opset11::Relu>(add_i);  // TAP POINT
        auto mul_i = std::make_shared<ov::opset11::Multiply>(relu_interior_i, scale_const);
        auto relu_boundary_i = std::make_shared<ov::opset11::Relu>(mul_i);  // boundary output
        m_nodes.push_back(add_i);
        m_nodes.push_back(relu_interior_i);
        m_nodes.push_back(mul_i);
        m_nodes.push_back(relu_boundary_i);
        set_name(add_i);
        set_name(relu_interior_i);
        set_name(mul_i);
        set_name(relu_boundary_i);

        if (head_set.count(i)) {
            // Cross-group edge from the interior Relu, mirroring Gemma4's
            // Multiply_1 → k/v_proj pattern.  Because relu_interior_i and
            // relu_boundary_i share the same metadesc, adding relu_interior_i
            // as a second output_layer leaves output_ometa = {"Relu…"} unchanged.
            auto kv_mm_i = std::make_shared<ov::opset11::MatMul>(relu_interior_i, kv_weight, false, false);
            m_nodes.push_back(kv_mm_i);
            set_name(kv_mm_i);
            kv_outputs.push_back(kv_mm_i->output(0));
        }

        current = relu_boundary_i;
    }

    // Non-repeated tail suffix.
    auto tail_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 2.f));
    auto tail_mul = std::make_shared<ov::opset11::Multiply>(current, tail_const);
    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_const);
    m_nodes.push_back(tail_const);
    m_nodes.push_back(tail_mul);
    m_nodes.push_back(tail_add);
    set_name(tail_const);
    set_name(tail_mul);
    set_name(tail_add);

    auto main_result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(main_result);
    set_name(main_result);
    ov::OutputVector outputs{main_result->output(0)};

    // Each head block's KV MatMul connects to an independent Result to avoid
    // creating asymmetric ov::Result consumer patterns across the repeated family.
    // If we accumulated kv_outputs into a single Result, only the last block would
    // indirectly connect to that Result via the accumulation chain, causing
    // isRegularResultCase to fail.
    for (auto&& kv_out : kv_outputs) {
        auto kv_result = std::make_shared<ov::opset11::Result>(kv_out);
        m_nodes.push_back(kv_result);
        set_name(kv_result);
        outputs.push_back(kv_result->output(0));
    }

    return std::make_shared<ov::Model>(outputs, ov::ParameterVector{input, kv_weight});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_multi_output_repeating_blocks(
    std::size_t repetitions,
    bool last_block_has_direct_result) {
    clear();
    if (repetitions == 0) {
        repetitions = 1;
    }

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    m_nodes.push_back(input);
    set_name(input);

    auto add_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto k_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {8});
    auto seed_indices = ov::opset11::Constant::create(ov::element::i32, ov::Shape{1, 1, 8}, {0, 1, 2, 3, 4, 5, 6, 7});
    auto tail_scale = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {0.5f});
    auto tail_bias = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {2.f});

    for (const auto& c : {add_const, k_const, seed_indices, tail_scale, tail_bias}) {
        m_nodes.push_back(c);
        set_name(c);
    }

    ov::Output<ov::Node> current_values = input;
    ov::Output<ov::Node> current_indices = seed_indices;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto indices_as_float = std::make_shared<ov::opset11::Convert>(current_indices, ov::element::f32);
        m_nodes.push_back(indices_as_float);
        set_name(indices_as_float);

        auto mixed = std::make_shared<ov::opset11::Add>(current_values, indices_as_float);
        m_nodes.push_back(mixed);
        set_name(mixed);

        auto shifted = std::make_shared<ov::opset11::Add>(mixed, add_const);
        m_nodes.push_back(shifted);
        set_name(shifted);

        auto topk = std::make_shared<ov::opset11::TopK>(shifted,
                                                        k_const,
                                                        -1,
                                                        ov::op::TopKMode::MAX,
                                                        ov::op::TopKSortType::SORT_VALUES,
                                                        ov::element::i32);
        m_nodes.push_back(topk);
        set_name(topk);

        current_values = topk->output(0);
        current_indices = topk->output(1);
    }

    auto tail_indices_as_float = std::make_shared<ov::opset11::Convert>(current_indices, ov::element::f32);
    m_nodes.push_back(tail_indices_as_float);
    set_name(tail_indices_as_float);

    auto tail_mixed = std::make_shared<ov::opset11::Add>(current_values, tail_indices_as_float);
    m_nodes.push_back(tail_mixed);
    set_name(tail_mixed);

    auto tail_mul = std::make_shared<ov::opset11::Multiply>(tail_mixed, tail_scale);
    m_nodes.push_back(tail_mul);
    set_name(tail_mul);

    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_bias);
    m_nodes.push_back(tail_add);
    set_name(tail_add);

    ov::OutputVector outputs;
    auto tail_result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(tail_result);
    set_name(tail_result);
    outputs.push_back(tail_result->output(0));

    if (last_block_has_direct_result) {
        auto direct_result = std::make_shared<ov::opset11::Result>(current_values);
        m_nodes.push_back(direct_result);
        set_name(direct_result);
        outputs.push_back(direct_result->output(0));
    }

    return std::make_shared<ov::Model>(outputs);
}

std::shared_ptr<ov::Node> ModelBuilder::get_block(const std::shared_ptr<ov::Node>& input) {
    std::vector<std::shared_ptr<ov::Node>> model_c(18, nullptr);
    model_c[0] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{0, 2, 1, 3});
    model_c[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
    model_c[2] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{2});
    model_c[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[5] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[6] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
    model_c[7] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[8] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[9] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
    model_c[10] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[11] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
    model_c[12] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[13] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[14] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[15] = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{40, 40});
    model_c[16] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
    model_c[17] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{1, 1, 40});

    for (const auto& c : model_c) {
        m_nodes.push_back(c);
        set_name(c);
    }

    std::vector<std::shared_ptr<ov::Node>> convert(3, nullptr);
    convert[0] = std::make_shared<ov::op::v0::Convert>(model_c[15], ov::element::f16);
    convert[1] = std::make_shared<ov::op::v0::Convert>(convert[0], ov::element::i32);
    convert[2] = std::make_shared<ov::op::v0::Convert>(model_c[12], ov::element::i32);

    for (const auto& c : convert) {
        m_nodes.push_back(c);
        set_name(c);
    }

    std::vector<std::shared_ptr<ov::Node>> op(16, nullptr);
    op[0] = std::make_shared<ov::op::v0::MatMul>(input, convert[1], false, true);
    op[1] = std::make_shared<ov::op::v1::Reshape>(op[0], model_c[16], false);
    op[2] = std::make_shared<ov::op::v1::Transpose>(op[1], model_c[0]);
    op[3] = std::make_shared<ov::op::v0::ShapeOf>(op[2]);
    op[4] = std::make_shared<ov::op::v1::Gather>(op[3], model_c[1], model_c[2]);
    op[5] = std::make_shared<ov::op::v1::Divide>(op[4], model_c[3], true);
    op[6] = std::make_shared<ov::op::v0::Floor>(op[5]);
    op[7] = std::make_shared<ov::op::v3::ScatterUpdate>(model_c[5], model_c[6], op[6], model_c[7]);
    op[8] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                       model_c[8],
                                                       op[7],
                                                       model_c[9],
                                                       std::vector<int64_t>{1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
    op[9] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                       op[7],
                                                       model_c[10],
                                                       model_c[11],
                                                       std::vector<int64_t>{1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
    op[10] = std::make_shared<ov::op::v1::Multiply>(op[9], convert[2]);
    op[11] = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{op[10], op[8]}, -1);
    op[12] = std::make_shared<ov::op::v1::Multiply>(model_c[13], op[11]);
    op[13] = std::make_shared<ov::op::v1::Multiply>(model_c[14], op[2]);
    op[14] = std::make_shared<ov::op::v1::Add>(op[13], op[12]);
    op[15] = std::make_shared<ov::op::v1::Reshape>(op[14], model_c[17], false);

    for (const auto& o : op) {
        m_nodes.push_back(o);
        set_name(o);
    }

    return op[15];
}

void ModelBuilder::set_name(const std::shared_ptr<ov::Node>& node) {
    node->set_friendly_name("node_" + std::to_string(m_name_idx++));
}

std::shared_ptr<ov::op::v0::Parameter> ModelBuilder::parameter(ov::element::Type type,
                                                               const ov::PartialShape& shape,
                                                               const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    return param;
}

void ModelBuilder::clear() {
    m_nodes.clear();
    m_sinks.clear();
    m_name_idx = 0;
}

ov::Output<ov::Node> ModelBuilder::setup_position_ids(LLMConfig& config, const ov::Output<ov::Node>& seq_source) {
    OPENVINO_ASSERT(!(config.internal_position_ids && config.position_ids.get_node()),
                    "internal_position_ids and position_ids are mutually exclusive");
    ov::Output<ov::Node> position_ids_output;

    if (config.internal_position_ids) {
        auto shape = std::make_shared<ov::opset11::ShapeOf>(seq_source, ov::element::i64);
        auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto seq_len = std::make_shared<ov::opset11::Gather>(shape, idx1, axis0);
        auto start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto range = std::make_shared<ov::op::v4::Range>(start, seq_len, step, ov::element::i64);
        range->set_friendly_name("model.internal_position_ids_range");
        auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(range, unsqueeze_axis);
        unsqueezed->set_friendly_name("model.internal_position_ids");
        position_ids_output = unsqueezed->output(0);
        // RoPE chain (Range → Unsqueeze → ... → Sin/Cos) matches NPUW's AddPositionIdsNode pattern
    } else if (config.position_ids.get_node()) {
        position_ids_output = config.position_ids;
    } else if (!config.rope) {
        position_ids_output = make_position_ids_2d();
    }
    // config.rope set without position_ids means RoPE was pre-built with position_ids baked in
    if (position_ids_output.get_node() && !config.rope) {
        config.rope = HalfRotationRoPE(config.head_dim, config.precision, position_ids_output);
    }

    return position_ids_output;
}

std::shared_ptr<ov::Model> ModelBuilder::make_model(const ov::Output<ov::Node>& output,
                                                    const std::string& result_name,
                                                    const std::string& model_name) {
    auto res = std::make_shared<ov::op::v0::Result>(output);
    res->set_friendly_name(result_name);
    res->output(0).set_names({result_name});

    return std::make_shared<ov::Model>(ov::OutputVector{res->output(0)}, m_sinks, model_name);
}

std::shared_ptr<ov::Model> ModelBuilder::build_llm(const LLMConfig& config_in) {
    clear();

    LLMConfig config = config_in;
    if (!config.norm)
        config.norm = LayerNorm(config.hidden_size, config.precision);
    if (!config.ffn) {
        if (config.num_experts > 0) {
            size_t moe_inter = config.moe_intermediate_size > 0
                                   ? config.moe_intermediate_size
                                   : config.intermediate_size;
            size_t moe_k = config.num_experts_per_tok > 0
                               ? config.num_experts_per_tok
                               : std::min<size_t>(2, config.num_experts);
            OPENVINO_ASSERT(moe_k >= 1 && moe_k <= config.num_experts,
                            "Invalid MoE config: num_experts_per_tok (",
                            moe_k, ") must be in [1, num_experts (", config.num_experts, ")]");
            config.ffn = MoEFFN(config.hidden_size, moe_inter, config.num_experts,
                                moe_k, config.precision);
        } else {
            config.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.precision, config.weight);
        }
    }
    const auto prec = config.precision;

    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");

    ov::Output<ov::Node> hidden_states;
    ov::Output<ov::Node> seq_source;

    if (config.use_inputs_embeds) {
        auto inputs_embeds =
            parameter(prec, ov::PartialShape{-1, -1, static_cast<int64_t>(config.hidden_size)}, "inputs_embeds");
        hidden_states = inputs_embeds->output(0);
        seq_source = inputs_embeds->output(0);
    } else {
        auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
        hidden_states =
            make_embedding(input_ids->output(0), config.vocab_size, config.hidden_size, "model.embed_tokens", prec);
        seq_source = input_ids->output(0);
    }

    setup_position_ids(config, seq_source);

    ov::Output<ov::Node> beam_idx_output;
    if (config.use_kv_cache) {
        auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");
        beam_idx_output = beam_idx->output(0);
    }

    auto sdpa_mask = make_causal_mask(seq_source, attention_mask->output(0), prec);

    // Shared GQA broadcast shape (embedding models only)
    ov::Output<ov::Node> shared_broadcast;
    if ((!config.use_kv_cache && !config.lm_head_weight) || config.force_gqa_broadcast) {
        shared_broadcast = make_shared_gqa_broadcast(attention_mask->output(0),
                                                     config.get_kv_heads(),
                                                     config.num_heads,
                                                     config.head_dim);
    }

    const auto hs = config.hidden_size;
    const auto kv_heads = config.get_kv_heads();

    Attention attn{};
    attn.hidden_size = hs;
    attn.num_heads = config.num_heads;
    attn.num_kv_heads = kv_heads;
    attn.head_dim = config.head_dim;
    attn.precision = prec;
    attn.weight_fn = config.weight;
    attn.bias_fn = config.attn_bias;
    attn.qk_norm = config.qk_norm;
    attn.rope_fn = config.rope;
    attn.sdpa_mask = sdpa_mask;
    attn.shared_broadcast_shape = shared_broadcast;

    if (config.use_kv_cache) {
        attn.kv_cache_fn = [&](const ov::Output<ov::Node>& k,
                               const ov::Output<ov::Node>& v,
                               size_t layer) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
            auto layer_str = std::to_string(layer);
            auto k_cache = make_kv_cache_concat(k,
                                                seq_source,
                                                beam_idx_output,
                                                kv_heads,
                                                config.head_dim,
                                                make_kv_var_id(layer_str, ".", "key"),
                                                prec);
            auto v_cache = make_kv_cache_concat(v,
                                                seq_source,
                                                beam_idx_output,
                                                kv_heads,
                                                config.head_dim,
                                                make_kv_var_id(layer_str, ".", "value"),
                                                prec);
            m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(k_cache.assign));
            m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(v_cache.assign));
            return {k_cache.concatenated, v_cache.concatenated};
        };
    }

    auto current =
        make_transformer_layers(hidden_states,
                                config.num_layers,
                                "model.layers.",
                                [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t layer) {
                                    if (config.pre_norm) {
                                        return make_pre_norm_layer(
                                            input,
                                            config.norm,
                                            [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                                                return attn(normed, {}, pfx, layer);
                                            },
                                            config.ffn,
                                            prefix);
                                    } else {
                                        return make_post_norm_layer(
                                            input,
                                            config.norm,
                                            [&](const ov::Output<ov::Node>& inp, const std::string& pfx) {
                                                return attn(inp, {}, pfx, layer);
                                            },
                                            config.ffn,
                                            prefix);
                                    }
                                });

    auto final_norm = config.norm(current, "model.norm");

    std::string model_name = "synthetic_decoder";

    if (config.lm_head_weight) {
        auto logits =
            make_lm_head(final_norm, config.hidden_size, config.vocab_size, "lm_head", prec, config.lm_head_weight);
        return make_model(logits, "logits", model_name);
    }
    return make_model(final_norm, "last_hidden_state", model_name);
}

std::shared_ptr<ov::Model> ModelBuilder::build_whisper_encoder(const WhisperConfig& config_in) {
    clear();
    WhisperConfig config = config_in;
    if (!config.norm)
        config.norm = LayerNorm(config.hidden_size, config.precision);
    if (!config.ffn)
        config.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.precision, config.weight);
    const auto prec = config.precision;
    const auto d = config.hidden_size;

    auto input_features = parameter(ov::element::f32,
                                    ov::PartialShape{-1,
                                                     static_cast<int64_t>(config.num_mel_bins),
                                                     static_cast<int64_t>(2 * config.max_source_positions)},
                                    "input_features");

    ov::Output<ov::Node> encoder_input = input_features->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(encoder_input, prec);
        cvt->set_friendly_name("model.encoder.input_convert");
        encoder_input = cvt->output(0);
    }

    auto conv1 = make_conv1d(encoder_input, config.num_mel_bins, d, 3, 1, 1, "model.encoder.conv1", prec);
    auto gelu1 = std::make_shared<ov::opset11::Gelu>(conv1);
    gelu1->set_friendly_name("model.encoder.conv1_gelu");

    auto conv2 = make_conv1d(gelu1->output(0), d, d, 3, 2, 1, "model.encoder.conv2", prec);
    auto gelu2 = std::make_shared<ov::opset11::Gelu>(conv2);
    gelu2->set_friendly_name("model.encoder.conv2_gelu");

    auto transpose_order = ov::opset11::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
    auto transposed = std::make_shared<ov::opset11::Transpose>(gelu2, transpose_order);
    transposed->set_friendly_name("model.encoder.transpose");

    auto pos_embed_val = fill_value_from_name("model.encoder.embed_positions.weight");
    auto pos_embed = ov::opset11::Constant::create(prec,
                                                   ov::Shape{1, config.max_source_positions, d},
                                                   std::vector<float>(config.max_source_positions * d, pos_embed_val));
    pos_embed->set_friendly_name("model.encoder.embed_positions.weight");

    auto embedded = std::make_shared<ov::opset11::Add>(transposed, pos_embed);
    embedded->set_friendly_name("model.encoder.pos_embed_add");

    Attention enc_attn{};
    enc_attn.hidden_size = d;
    enc_attn.num_heads = config.num_heads;
    enc_attn.num_kv_heads = config.num_heads;
    enc_attn.head_dim = config.head_dim;
    enc_attn.precision = prec;
    enc_attn.weight_fn = config.weight;
    enc_attn.bias_fn = config.attn_bias;
    enc_attn.o_proj_name = "self_attn.out_proj";

    auto current =
        make_transformer_layers(embedded->output(0),
                                config.get_encoder_layers(),
                                "model.encoder.layers.",
                                [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t /*layer*/) {
                                    return make_pre_norm_layer(
                                        input,
                                        config.norm,
                                        [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                                            return enc_attn(normed, {}, pfx);
                                        },
                                        config.ffn,
                                        prefix);
                                });

    auto final_norm = config.norm(current, "model.encoder.layer_norm");

    // Always f32 output — WhisperPipeline reads encoder output as f32
    ov::Output<ov::Node> encoder_output = final_norm;
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(final_norm, ov::element::f32);
        cvt->set_friendly_name("model.encoder.output_convert");
        encoder_output = cvt->output(0);
    }

    return make_model(encoder_output, "last_hidden_state", "synthetic_whisper_encoder");
}


std::shared_ptr<ov::Model> ModelBuilder::build_whisper_decoder(const WhisperConfig& config_in) {
    clear();
    WhisperConfig config = config_in;
    if (!config.norm)
        config.norm = LayerNorm(config.hidden_size, config.precision);
    if (!config.ffn)
        config.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.precision, config.weight);
    const auto prec = config.precision;
    const auto d = config.hidden_size;
    const auto heads = config.num_heads;
    const auto hd = config.head_dim;

    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto encoder_hidden_states =
        parameter(ov::element::f32,
                  ov::PartialShape{-1, static_cast<int64_t>(config.get_encoder_seq_len()), static_cast<int64_t>(d)},
                  "encoder_hidden_states");
    auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    ov::Output<ov::Node> enc_hs = encoder_hidden_states->output(0);
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(enc_hs, prec);
        cvt->set_friendly_name("model.decoder.enc_hs_convert");
        enc_hs = cvt->output(0);
    }

    auto token_embed = make_embedding(input_ids->output(0), config.vocab_size, d, "model.decoder.embed_tokens", prec);

    // Pre-build layer 0's key read state — NPUW matchers need kv_seq_len from ShapeOf(beam_gather)[2]
    auto layer0_k_read = make_kv_cache_read(input_ids->output(0),
                                            beam_idx->output(0),
                                            heads,
                                            hd,
                                            make_kv_var_id("0", ".decoder.", "key"),
                                            prec);

    auto cache_pos = make_cache_position_ids(input_ids->output(0), layer0_k_read.beam_gather, "model.decoder.");
    auto hidden_states = make_learned_positional_embedding(token_embed,
                                                           cache_pos.position_ids,
                                                           config.max_target_positions,
                                                           d,
                                                           prec,
                                                           "model.decoder.");
    auto shared_mask = make_whisper_causal_mask(cache_pos, "model.decoder.");

    // Self-attention (layer-0 reuses pre-built key Variable)
    Attention self_attn{};
    self_attn.hidden_size = d;
    self_attn.num_heads = heads;
    self_attn.num_kv_heads = heads;
    self_attn.head_dim = hd;
    self_attn.precision = prec;
    self_attn.weight_fn = config.weight;
    self_attn.bias_fn = config.attn_bias;
    self_attn.o_proj_name = "self_attn.out_proj";
    self_attn.sdpa_mask = shared_mask;

    self_attn.kv_cache_fn = [&](const ov::Output<ov::Node>& k,
                                const ov::Output<ov::Node>& v,
                                size_t layer) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
        auto layer_str = std::to_string(layer);
        KVCacheResult k_cache;
        if (layer == 0) {
            auto k_var_id = make_kv_var_id(layer_str, ".decoder.", "key");
            auto k_concat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{layer0_k_read.beam_gather, k}, 2);
            k_concat->set_friendly_name(k_var_id + "_concat");
            auto k_assign = std::make_shared<ov::op::v6::Assign>(k_concat, layer0_k_read.variable);
            k_assign->set_friendly_name(k_var_id + "_assign");
            k_cache = {k_concat->output(0), layer0_k_read.beam_gather, k_assign};
        } else {
            k_cache = make_kv_cache_concat(k,
                                           input_ids->output(0),
                                           beam_idx->output(0),
                                           heads,
                                           hd,
                                           make_kv_var_id(layer_str, ".decoder.", "key"),
                                           prec);
        }
        auto v_cache = make_kv_cache_concat(v,
                                            input_ids->output(0),
                                            beam_idx->output(0),
                                            heads,
                                            hd,
                                            make_kv_var_id(layer_str, ".decoder.", "value"),
                                            prec);
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(k_cache.assign));
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(v_cache.assign));
        return {k_cache.concatenated, v_cache.concatenated};
    };

    // Cross-attention (store-only encoder KV cache)
    Attention cross_attn{};
    cross_attn.hidden_size = d;
    cross_attn.num_heads = heads;
    cross_attn.num_kv_heads = heads;
    cross_attn.head_dim = hd;
    cross_attn.precision = prec;
    cross_attn.weight_fn = config.weight;
    cross_attn.bias_fn = config.attn_bias;
    cross_attn.o_proj_name = "encoder_attn.out_proj";
    cross_attn.attn_prefix = "encoder_attn.";

    cross_attn.kv_cache_fn = [&](const ov::Output<ov::Node>& k,
                                 const ov::Output<ov::Node>& v,
                                 size_t layer) -> std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> {
        auto layer_str = std::to_string(layer);
        auto k_cache = make_encoder_kv_cache(k, heads, hd, make_kv_var_id(layer_str, ".encoder.", "key"), prec);
        auto v_cache = make_encoder_kv_cache(v, heads, hd, make_kv_var_id(layer_str, ".encoder.", "value"), prec);
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(k_cache.assign));
        m_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(v_cache.assign));
        return {k_cache.concatenated, v_cache.concatenated};
    };

    auto current = make_transformer_layers(
        hidden_states,
        config.get_decoder_layers(),
        "model.decoder.layers.",
        [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t layer) {
            auto call_self = [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                return self_attn(normed, {}, pfx, layer);
            };
            auto call_cross = [&](const ov::Output<ov::Node>& normed, const std::string& pfx) {
                return cross_attn(normed, enc_hs, pfx, layer);
            };

            return make_cross_attn_decoder_layer(input, config.norm, call_self, call_cross, config.ffn, prefix);
        });

    auto final_norm = config.norm(current, "model.decoder.layer_norm");
    auto logits = make_lm_head(final_norm, d, config.vocab_size, "proj_out", prec, config.weight);

    // Always f32 output — WhisperPipeline reads logits as f32
    ov::Output<ov::Node> logits_out = logits;
    if (prec != ov::element::f32) {
        auto cvt = std::make_shared<ov::op::v0::Convert>(logits, ov::element::f32);
        cvt->set_friendly_name("model.decoder.logits_convert");
        logits_out = cvt->output(0);
    }

    return make_model(logits_out, "logits", "synthetic_whisper_decoder");
}

std::shared_ptr<ov::Model> ModelBuilder::build_embedding_encoder(const BertConfig& config_in) {
    clear();
    BertConfig config = config_in;
    if (!config.norm)
        config.norm = LayerNorm(config.hidden_size, config.precision);
    if (!config.ffn)
        config.ffn = SwiGLU(config.hidden_size, config.intermediate_size, config.precision, config.weight);

    const auto prec = config.precision;
    const auto hs = config.hidden_size;

    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");
    auto token_type_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "token_type_ids");

    auto word_embed = make_embedding(input_ids->output(0), config.vocab_size, hs, "embeddings.word_embeddings", prec);

    auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
    auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto axis0 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto seq_len = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, axis0);
    auto start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto pos_ids = std::make_shared<ov::op::v4::Range>(start, seq_len, step, ov::element::i64);
    pos_ids->set_friendly_name("embeddings.position_ids");
    auto pos_embed =
        make_embedding(pos_ids->output(0), config.max_position_embeddings, hs, "embeddings.position_embeddings", prec);

    auto type_embed =
        make_embedding(token_type_ids->output(0), config.type_vocab_size, hs, "embeddings.token_type_embeddings", prec);

    auto embed_sum1 = std::make_shared<ov::opset11::Add>(word_embed, pos_embed);
    embed_sum1->set_friendly_name("embeddings.add_pos");
    auto embed_sum2 = std::make_shared<ov::opset11::Add>(embed_sum1, type_embed);
    embed_sum2->set_friendly_name("embeddings.add_type");
    auto embed_normed = config.norm(embed_sum2->output(0), "embeddings.LayerNorm");

    auto sdpa_mask = make_padding_mask(attention_mask->output(0), prec);

    Attention bert_attn{};
    bert_attn.hidden_size = hs;
    bert_attn.num_heads = config.num_heads;
    bert_attn.num_kv_heads = config.num_heads;
    bert_attn.head_dim = config.head_dim;
    bert_attn.precision = prec;
    bert_attn.weight_fn = config.weight;
    bert_attn.bias_fn = config.attn_bias;
    bert_attn.sdpa_mask = sdpa_mask;

    auto current =
        make_transformer_layers(embed_normed,
                                config.num_layers,
                                "encoder.layer.",
                                [&](const ov::Output<ov::Node>& input, const std::string& prefix, size_t /*layer*/) {
                                    return make_post_norm_layer(
                                        input,
                                        config.norm,
                                        [&](const ov::Output<ov::Node>& inp, const std::string& pfx) {
                                            return bert_attn(inp, {}, pfx);
                                        },
                                        config.ffn,
                                        prefix);
                                });

    return make_model(current, "last_hidden_state", "synthetic_encoder_model");
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
