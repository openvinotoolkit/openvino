// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inkling_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "builder/api/metadata_store.hpp"
#include "builder/graph_emitter.hpp"
#include "builder/metadata_checks.hpp"
#include "openvino/opsets/opset13.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace ov::frontend::gguf {
namespace {
namespace op = ov::opset13;
using Value = ov::Output<ov::Node>;

auto ints(const std::vector<int64_t>& values) {
    return op::Constant::create(ov::element::i64, {values.size()}, values);
}

auto scalar(float value) {
    return op::Constant::create(ov::element::f32, {}, {value});
}

Value reshape(const Value& value, const std::vector<int64_t>& shape, bool special_zero = false) {
    return std::make_shared<op::Reshape>(value, ints(shape), special_zero);
}

Value transpose(const Value& value, const std::vector<int64_t>& order) {
    return std::make_shared<op::Transpose>(value, ints(order));
}

Value slice(const Value& value, int64_t axis, int64_t start, int64_t length) {
    const auto end = length < 0 ? std::numeric_limits<int64_t>::max() : start + length;
    return std::make_shared<op::Slice>(value, ints({start}), ints({end}), ints({1}), ints({axis}));
}

Value add(const Value& a, const Value& b) {
    return std::make_shared<op::Add>(a, b);
}

Value mul(const Value& a, const Value& b) {
    return std::make_shared<op::Multiply>(a, b);
}

Value softcap(const Value& value, float cap) {
    return mul(std::make_shared<op::Tanh>(mul(value, scalar(1.f / cap))), scalar(cap));
}

class InklingBuilder : public ModelBuilder {
public:
    explicit InklingBuilder(const BuildContext& context)
        : meta(context.metadata),
          arch(context.arch),
          e(context.weights->weights, context.arch, context.weights->translators) {}

    std::shared_ptr<GgufGraph> build() override {
        layers = positive("block_count");
        width = positive("embedding_length");
        kernel = positive("shortconv.l_cache");
        block_size = positive("attn_res.block_size");
        relative = positive("attention.relative_length");
        extent = positive("attention.relative_extent");
        window = positive("attention.sliding_window");
        const auto vocab = positive("vocab_size");
        const auto unpadded = positive("vocab_size_unpadded");
        experts = positive("expert_count");
        eps = positive_float("attention.layer_norm_rms_epsilon");
        qk_clip = positive_float("attention.qk_clip");
        rel_clip = positive_float("attention.relative_clip");
        gate_cap = positive_float("activation.situ_beta");
        up_cap = positive_float("activation.situ_linear_beta");
        const auto scale = meta.get_float(arch + ".expert_weights_scale").value_or(1.);
        const auto floor = positive("attention.log_scale_floor");
        const auto alpha = meta.get_float(arch + ".attention.log_scale_alpha");
        const auto final_cap = meta.get_float(arch + ".final_logit_softcapping").value_or(0.);
        OPENVINO_ASSERT(layers <= 1024 && kernel >= 2 && kernel <= 64 && unpadded <= vocab,
                        "[GGUF] invalid Inkling dimensions");
        OPENVINO_ASSERT(representable_float(scale) && alpha && representable_float(*alpha) && *alpha >= 0,
                        "[GGUF] invalid Inkling scaling parameters");
        OPENVINO_ASSERT(representable_float(final_cap) && final_cap >= 0 && std::isfinite(1.f / gate_cap) &&
                            std::isfinite(1.f / up_cap) &&
                            (final_cap == 0 || std::isfinite(1.f / static_cast<float>(final_cap))),
                        "[GGUF] invalid Inkling clipping parameters");
        route_scale = scale == 0 ? 1.f : static_cast<float>(scale);
        sliding = meta.get_int_array(arch + ".attention.sliding_window_pattern");
        const auto is_flag = [](int64_t flag) {
            return flag == 0 || flag == 1;
        };
        OPENVINO_ASSERT(
            sliding.size() == static_cast<size_t>(layers) && std::all_of(sliding.begin(), sliding.end(), is_flag),
            "[GGUF] Inkling requires a sliding-window flag for every physical layer");
        int64_t start = 0, end = layers, passes = 1;
        if (arch == "inkling2_loop") {
            start = positive("loop.start");
            end = positive("loop.end");
            passes = positive("loop.count");
            const auto rank = positive("loop.gate_rank");
            const bool varying = meta.get_bool(arch + ".loop.varying").value_or(false);
            OPENVINO_ASSERT(start < end && end <= layers && start % block_size == 0 && end % block_size == 0 &&
                                passes >= (varying ? 1 : 2) && passes <= 1024 &&
                                layers + (passes - 1) * (end - start) <= 1024,
                            "[GGUF] invalid Inkling loop core, pass count or cache-slot count");
            for (auto layer = start; layer < end; ++layer) {
                const auto p = "blk." + std::to_string(layer) + ".loop_";
                weight(p + "msg_query.weight", {4, width});
                for (const auto* ctrl : {"q", "k", "v", "ffn"}) {
                    weight(p + ctrl + "_ctrl_down.weight", {rank, 2 * width});
                    weight(p + ctrl + "_ctrl_gain.weight", {width, rank});
                    weight(p + ctrl + "_ctrl_forget.weight", {width, rank});
                }
            }
        }
        auto& graph = *e.graph();
        graph.swa_window_size = static_cast<int>(window);
        graph.model_contract.batchable_recurrent = false;
        auto positions = e.add_input("inp_pos", ov::element::i32, ps({1, 1, 1, -1}));
        e.add_input("inp_tokens", ov::element::i32, ps({1, 1, 1, -1}));
        e.add_input("inp_out_ids", ov::element::i32, ps({1, 1, 1, -1}));
        e.add_input("inp_kv_idx", ov::element::i32, ps({1, 1, 1, -1}));
        e.add_input("self_kq_mask", ov::element::f32, ps({1, 1, -1, -1}));
        e.add_input("self_kq_mask_swa", ov::element::f32, ps({1, 1, -1, -1}));
        e.add_input("token_len_per_seq", ov::element::i64, ps({1}));
        e.add_input("cache_positions", ov::element::i32, ps({1, -1, 1, 1}));
        e.add_op("GGML_OP_SET_ROWS",
                 "cache_positions",
                 {save(reshape(positions, {1, -1, 1, 1})), "inp_kv_idx", "cache_positions"});
        graph.model_output_names.push_back("cache_positions");
        auto position = std::make_shared<op::Convert>(positions, ov::element::f32);
        Value ratio = mul(add(position, scalar(1.f)), scalar(1.f / floor));
        ratio = std::make_shared<op::Maximum>(ratio, scalar(1.f));
        tau = reshape(add(mul(std::make_shared<op::Log>(ratio), scalar(static_cast<float>(*alpha))), scalar(1.f)),
                      {0, -1, 1, 1},
                      true);
        auto keys = std::make_shared<op::Convert>(e.value("cache_positions"), ov::element::i64);
        auto distance = std::make_shared<op::Subtract>(
            reshape(std::make_shared<op::Convert>(positions, ov::element::i64), {0, 1, -1, 1}, true),
            reshape(keys, {0, 1, 1, -1}, true));
        auto distance_shape = std::make_shared<op::ShapeOf>(distance);
        distance_batch = std::make_shared<op::Gather>(distance_shape, ints({0}), ints({0}));
        distance_rows_cols = std::make_shared<op::Gather>(distance_shape, ints({2, 3}), ints({0}));
        for (const bool swa : {false, true}) {
            const auto span = ints({swa ? window : extent});
            auto valid = std::make_shared<op::LogicalAnd>(std::make_shared<op::GreaterEqual>(distance, ints({0})),
                                                          std::make_shared<op::Less>(distance, span));
            relative_indices[swa] = std::make_shared<op::Select>(valid, distance, span);
            // Halving keeps masked logits finite in f16 after the bias, sink and score additions.
            half_masks[swa] = mul(e.value(swa ? "self_kq_mask_swa" : "self_kq_mask"), scalar(0.5f));
        }
        auto cur = operation("GGML_OP_GET_ROWS", {weight("token_embd.weight", {vocab, width}), e.value("inp_tokens")});
        cur = norm(cur, weight("token_embd_norm.weight", {width}), eps);
        // Encoded image embeddings already live at this normalized token boundary.
        cur.get_node_shared_ptr()->get_rt_info()["gguf.token_embedding"] = true;
        std::vector<Value> blocks;
        for (int64_t layer = 0; layer < start; ++layer)
            cur = build_layer(layer, layer, cur, blocks);
        const auto prelude = cur;
        const auto prelude_blocks = blocks;
        for (int64_t pass = 0; pass < passes; ++pass) {
            std::vector<Value> bank;
            Value bank_out;
            if (pass > 0) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank_out = cur;
                cur = prelude;
                blocks = prelude_blocks;
            }
            for (auto layer = start; layer < end; ++layer) {
                const auto slot = pass == 0 ? layer : layers + (pass - 1) * (end - start) + layer - start;
                Value messages;
                if (pass > 0)
                    messages = mix(bank, bank_out, weight("blk." + std::to_string(layer) + ".loop_msg_query.weight"));
                cur = build_layer(layer, slot, cur, blocks, messages);
            }
        }
        for (auto layer = end; layer < layers; ++layer)
            cur = build_layer(layer, layer, cur, blocks);
        cur = token_rows(mix(blocks, cur, weight("output_res_query.weight", {width})));
        cur = norm(cur, weight("output_norm.weight", {width}), eps);
        cur = operation("GGML_OP_GET_ROWS", {cur, e.value("inp_out_ids")});
        cur = mm(weight("output.weight", {vocab, width}), cur);
        if (final_cap > 0)
            cur = softcap(cur, static_cast<float>(final_cap));
        cur = slice(cur, 3, 0, unpadded);
        graph.model_output_names.insert(graph.model_output_names.begin(), save(cur));
        return e.graph();
    }

private:
    int64_t positive(const std::string& suffix) const {
        return require_positive_int(meta, arch + "." + suffix, "Inkling");
    }

    float positive_float(const std::string& suffix) const {
        return require_positive_float(meta, arch + "." + suffix, "Inkling");
    }

    int64_t per_layer(const std::string& suffix, int64_t layer) const {
        const auto key = arch + "." + suffix;
        if (meta.get_int(key))
            return positive(suffix);
        const auto array = meta.get_int_array(key);
        OPENVINO_ASSERT(array.size() == static_cast<size_t>(layers) && array[layer] > 0 &&
                            array[layer] <= std::numeric_limits<int>::max(),
                        "[GGUF] invalid Inkling per-layer dimensions: ",
                        key);
        return array[layer];
    }

    std::string next_name() {
        return "inkling." + std::to_string(serial++);
    }

    std::string save(const Value& value) {
        auto name = next_name();
        (*e.graph()->values)[name] = value;
        return name;
    }

    Value operation(const std::string& type,
                    const std::vector<Value>& inputs,
                    int op_case = 0,
                    const std::map<std::string, ov::Any>& attrs = {}) {
        std::vector<std::string> names;
        for (const auto& input : inputs)
            names.push_back(save(input));
        const auto name = next_name();
        e.add_op(type, name, names, op_case, attrs);
        return e.value(name);
    }

    Value weight(const std::string& name, const std::vector<int64_t>& shape = {}) {
        if (!shape.empty()) {
            const auto squeeze_front = [](ov::Shape s) {
                while (s.size() > 1 && s.front() == 1)
                    s.erase(s.begin());
                return s;
            };
            const auto actual = e.weight_shape(name);
            OPENVINO_ASSERT(squeeze_front(actual) == squeeze_front(ov::Shape(shape.begin(), shape.end())),
                            "[GGUF] invalid Inkling tensor shape: ",
                            name,
                            "; expected ",
                            ov::PartialShape(shape),
                            ", got ",
                            ov::PartialShape(actual));
        }
        e.add_weight(name);
        return e.value(name);
    }

    Value mm(const Value& w, const Value& value) {
        return operation("GGML_OP_MUL_MAT", {w, value});
    }

    Value norm(const Value& value, const Value& w, float epsilon) {
        auto normalized = operation("GGML_OP_RMS_NORM", {value}, 0, {{"eps", epsilon}});
        return w.get_node() ? mul(normalized, w) : normalized;
    }

    Value clip(Value value, const Value& scale, float threshold) {
        if (scale.get_node())
            value = mul(value, scale);
        // No epsilon: it only matters below the threshold, where the factor clamps to 1, and an epsilon
        // denominator or a sum of squares would pin this chain to f32 on GPU.
        auto factor =
            std::make_shared<op::Divide>(scalar(threshold), std::make_shared<op::ReduceL2>(value, ints({3}), true));
        return mul(value, std::make_shared<op::Clamp>(factor, 0., 1.));
    }

    Value token_rows(const Value& value, int64_t row = -1) {
        return reshape(row < 0 ? value : slice(value, 2, row, 1), {0, 1, -1, width}, true);
    }

    // Blocks are stored as normalized [B, T, 1, width] rows so each one is normalized only once.
    Value stack_row(const Value& value) {
        return norm(reshape(value, {0, -1, 1, width}, true), {}, 1e-6f);
    }

    Value mix(const std::vector<Value>& blocks, const Value& cur, const Value& query) {
        ov::OutputVector states(blocks.rbegin(), blocks.rend());
        states.push_back(stack_row(cur));
        const auto values = std::make_shared<op::Concat>(states, 2);
        const auto queries = reshape(query, {1, 1, -1, width});
        auto scores = std::make_shared<op::MatMul>(queries, values, false, true);
        auto probabilities = std::make_shared<op::Softmax>(mul(scores, scalar(1.f / std::sqrt(width))), 3);
        return std::make_shared<op::MatMul>(probabilities, values);
    }

    Value control(const Value& cur, const Value& message, const std::string& prefix) {
        auto joined = std::make_shared<op::Concat>(ov::OutputVector{norm(cur, {}, 1e-6f), norm(message, {}, 1e-6f)}, 3);
        auto gate = mm(weight(prefix + "_ctrl_down.weight"), joined);
        auto gain = std::make_shared<op::Tanh>(mm(weight(prefix + "_ctrl_gain.weight"), gate));
        auto forget = std::make_shared<op::Tanh>(mm(weight(prefix + "_ctrl_forget.weight"), gate));
        return add(std::make_shared<op::Subtract>(cur, mul(forget, cur)), mul(gain, message));
    }

    Value convolve(const Value& cur, const Value& w, const std::string& name, const Value& dynamic = {}) {
        const auto channels = cur.get_partial_shape()[3].get_length();
        // Dynamic taps read the lagged inputs as extra one-hot channels of the same causal convolution.
        const auto copies = dynamic.get_node() ? kernel + 1 : 1;
        auto input = cur;
        auto filters = w;
        if (copies > 1) {
            input = std::make_shared<op::Concat>(ov::OutputVector(static_cast<size_t>(copies), cur), 3);
            std::vector<float> one_hot(static_cast<size_t>(kernel * channels * kernel), 0.f);
            for (int64_t lag = 0; lag < kernel; ++lag)
                for (int64_t channel = 0; channel < channels; ++channel)
                    one_hot[static_cast<size_t>((lag * channels + channel) * kernel + kernel - 1 - lag)] = 1.f;
            const auto one_hot_shape = ov::Shape{static_cast<size_t>(kernel * channels), static_cast<size_t>(kernel)};
            filters = std::make_shared<op::Concat>(
                ov::OutputVector{reshape(w, {channels, kernel}),
                                 op::Constant::create(w.get_element_type(), one_hot_shape, one_hot)},
                0);
        }
        const auto state = e.add_input(name, ov::element::f32, ps({1, 1, channels * copies, kernel - 1}));
        auto history = std::make_shared<op::Concat>(ov::OutputVector{state, transpose(input, {0, 1, 3, 2})}, 3);
        const auto update = save(slice(history, 3, -(kernel - 1), -1));
        e.graph()->recurrent_states.emplace_back(name, update);
        e.graph()->model_output_names.push_back(update);
        Value out = operation("GGML_OP_SSM_CONV", {history, filters}, 0, {{"batch_major", true}});
        if (copies > 1) {
            const auto heads = dynamic.get_partial_shape()[3].get_length() / kernel;
            auto taps = reshape(dynamic, {0, -1, heads, kernel}, true);
            const auto conv = out;
            out = slice(conv, 3, 0, channels);
            for (int64_t lag = 0; lag < kernel; ++lag) {
                auto rows =
                    reshape(slice(conv, 3, (lag + 1) * channels, channels), {0, -1, heads, channels / heads}, true);
                out = add(out, reshape(mul(rows, slice(taps, 3, lag, 1)), {0, 1, -1, channels}, true));
            }
        }
        return mul(add(cur, out), scalar(1.f / std::sqrt((copies > 1 ? 2 : 1) * kernel + 1.f)));
    }

    Value cache(const Value& value, const std::string& name, int64_t heads, int64_t head) {
        e.add_input(name, ov::element::f16, ps({1, -1, heads, head}));
        e.add_op("GGML_OP_SET_ROWS", name, {save(value), "inp_kv_idx", name});
        e.graph()->model_output_names.push_back(name);
        return e.value(name);
    }

    Value attention(const std::vector<Value>& inputs, const std::string& p, int64_t layer, int64_t slot) {
        const bool swa = sliding[layer] != 0;
        const auto heads = per_layer("attention.head_count", layer);
        const auto kv_heads = per_layer("attention.head_count_kv", layer);
        const auto head_key =
            swa && meta.has(arch + ".attention.key_length_swa") ? "attention.key_length_swa" : "attention.key_length";
        const auto head = positive(head_key);
        const auto value_key = swa && meta.has(arch + ".attention.value_length_swa") ? "attention.value_length_swa"
                                                                                     : "attention.value_length";
        const auto value_head = meta.has(arch + "." + value_key) ? positive(value_key) : head;
        OPENVINO_ASSERT(heads % kv_heads == 0 && head == value_head,
                        "[GGUF] Inkling requires divisible query/KV head counts and equal key/value head widths");
        const auto kv_width = head * kv_heads;
        const auto state = "conv_state_l" + std::to_string(slot);
        const auto conv_kv = [&](const std::string& n, const Value& in) {
            const auto prefix = p + "attn_" + n;
            return convolve(mm(weight(prefix + ".weight", {kv_width, width}), in),
                            weight(prefix + "_conv.weight", {kv_width, kernel}),
                            state + "_" + n,
                            swa ? Value{} : mm(weight(prefix + "_conv_proj.weight", {kernel * kv_heads, width}), in));
        };
        auto q = mm(weight(p + "attn_q.weight", {head * heads, width}), inputs[0]);
        auto k = conv_kv("k", inputs[1]);
        auto v = conv_kv("v", inputs[2]);
        q = clip(reshape(q, {0, -1, heads, head}, true), weight(p + "attn_q_clip.weight", {heads, head}), qk_clip);
        k = clip(reshape(k, {0, -1, kv_heads, head}, true), {}, qk_clip);
        v = norm(reshape(v, {0, -1, kv_heads, head}, true), {}, eps);
        auto r = clip(reshape(mm(weight(p + "attn_rel.weight", {relative * heads, width}), inputs[0]),
                              {0, -1, heads, relative},
                              true),
                      weight(p + "attn_rel_clip.weight", {heads, relative}),
                      rel_clip);
        if (!swa) {
            q = mul(q, tau);
            r = mul(r, tau);
        }
        k = cache(k, "cache_k_l" + std::to_string(slot), kv_heads, head);
        v = cache(v, "cache_v_l" + std::to_string(slot), kv_heads, head);
        const auto span = swa ? window : extent;
        auto padded =
            std::make_shared<op::Pad>(reshape(weight(p + "attn_rel_bias.weight", {span, relative}), {span, relative}),
                                      ints({0, 0}),
                                      ints({1, 0}),
                                      ov::op::PadMode::CONSTANT);
        auto table = std::make_shared<op::MatMul>(transpose(r, {0, 2, 1, 3}), padded, false, true);
        auto bias_shape =
            std::make_shared<op::Concat>(ov::OutputVector{distance_batch, ints({heads}), distance_rows_cols}, 0);
        Value bias =
            std::make_shared<op::GatherElements>(table,
                                                 std::make_shared<op::Broadcast>(relative_indices[swa], bias_shape),
                                                 3);
        auto sinks = mm(weight(p + "attn_sink_proj.weight", {heads, width}), inputs[0]);
        bias = std::make_shared<op::Subtract>(bias, transpose(sinks, {0, 3, 2, 1}));
        bias = add(bias, half_masks[swa]);
        // The zero sink represents the removed query-dependent sink logit.
        auto sinks_zero = op::Constant::create(ov::element::f32, {1, 1, 1, static_cast<size_t>(heads)}, {0.f});
        auto out =
            operation("GGML_OP_FLASH_ATTN_EXT",
                      {q, k, v, bias, sinks_zero},
                      100,
                      {{"scale", 1.f / std::sqrt(static_cast<float>(head))}, {"is_swa", swa}, {"explicit_mask", true}});
        out = reshape(out, {0, 1, -1, heads * head}, true);
        out = mul(out,
                  std::make_shared<op::Sigmoid>(mm(weight(p + "attn_gate.weight", {heads * head, width}), inputs[0])));
        out = mm(weight(p + "attn_output.weight", {width, heads * head}), out);
        return convolve(out, weight(p + "attn_post_conv.weight", {width, kernel}), state + "_attn");
    }

    Value ffn(const Value& cur, const std::string& p, int64_t layer) {
        const auto hidden = per_layer("expert_feed_forward_length", layer);
        const auto used = per_layer("expert_used_count", layer);
        OPENVINO_ASSERT(used <= experts, "[GGUF] Inkling expert_used_count exceeds expert_count");
        auto probs = std::make_shared<op::Sigmoid>(mm(weight(p + "ffn_gate_inp.weight", {experts, width}), cur));
        Value selection = probs;
        if (e.has_weight(p + "exp_probs_b.bias"))
            selection = add(probs, weight(p + "exp_probs_b.bias", {experts}));
        auto selected = operation("GGML_OP_TOP_K", {selection}, 0, {{"k", used}});
        auto weights = operation("GGML_OP_GET_ROWS", {probs, selected}, 10);
        // Same floor as llama.cpp; Clamp rather than Maximum keeps this chain out of GPU f32 islands.
        auto sum = std::make_shared<op::Clamp>(std::make_shared<op::ReduceSum>(weights, ints({2}), true),
                                               6.103515625e-5,
                                               std::numeric_limits<double>::infinity());
        weights = mul(weights, std::make_shared<op::Divide>(scalar(route_scale), sum));
        // Expert-major [K, T, *] throughout: the activation is elementwise, so neither GatherMatmul
        // output needs a transpose back to token-major.
        auto ids = std::make_shared<op::Convert>(reshape(selected, {-1, used}), ov::element::i32);
        const auto experts_value = [&](const std::string& name, const std::vector<int64_t>& shape) {
            Value w = weight(name, shape);
            if (w.get_partial_shape().size() == 4)
                w = reshape(w, {shape.begin() + 1, shape.end()});
            return w;
        };
        const auto tokens = reshape(cur, {1, -1, width});
        Value joint = std::make_shared<ov::op::internal::GatherMatmul>(
            tokens,
            experts_value(p + "ffn_gate_up_exps.weight", {1, experts, 2 * hidden, width}),
            ids);
        auto gate = slice(joint, 2, 0, hidden);
        auto up = slice(joint, 2, hidden, hidden);
        auto activation = mul(mul(softcap(gate, gate_cap), std::make_shared<op::Sigmoid>(gate)), softcap(up, up_cap));
        Value out = std::make_shared<ov::op::internal::GatherMatmul>(
            activation,
            experts_value(p + "ffn_down_exps.weight", {1, experts, width, hidden}),
            ids);
        if (e.has_weight(p + "ffn_down_exps.scale")) {
            auto scales = operation("GGML_OP_GET_ROWS",
                                    {reshape(weight(p + "ffn_down_exps.scale"), {1, 1, 1, experts}), selected},
                                    10,
                                    {{"shared_row", true}});
            weights = mul(weights, scales);
        }
        auto slot_weights = std::make_shared<op::Unsqueeze>(transpose(reshape(weights, {-1, used}), {1, 0}), ints({2}));
        auto combined = std::make_shared<op::ReduceSum>(mul(out, slot_weights), ints({0}), false);
        return std::make_shared<op::Reshape>(combined, std::make_shared<op::ShapeOf>(cur), false);
    }

    Value build_layer(int64_t layer,
                      int64_t slot,
                      const Value& cur,
                      std::vector<Value>& blocks,
                      const Value& messages = {}) {
        const auto p = "blk." + std::to_string(layer) + ".";
        const bool swa = sliding[layer] != 0;
        const bool loop = messages.get_node() != nullptr;
        Value mixed;
        if (layer > 0)
            mixed = mix(blocks, cur, weight(p + "attn_res_query.weight", {swa ? 1 : 3, width}));
        const std::string controls[] = {"q", "k", "v"};
        std::vector<Value> inputs(3);
        for (int j = 0; j < 3; ++j) {
            // Without loop controls, SWA layers feed one shared normalized input to Q, K and V.
            if (swa && !loop && j > 0) {
                inputs[j] = inputs[0];
                continue;
            }
            auto x = mixed.get_node() ? token_rows(mixed, swa ? 0 : j) : cur;
            if (loop)
                x = control(x, token_rows(messages, j), p + "loop_" + controls[j]);
            inputs[j] =
                norm(x,
                     weight(p + (swa ? "attn_norm.weight" : "attn_" + controls[j] + "_inp_norm.weight"), {width}),
                     eps);
        }
        auto out = attention(inputs, p, layer, slot);
        // Blocks are indexed by physical layer, not by the unrolled cache slot.
        if (layer % block_size == 0)
            blocks.push_back(stack_row(cur));
        else
            out = add(cur, out);
        auto residual = out;
        out = token_rows(mix(blocks, out, weight(p + "ffn_res_query.weight", {width})));
        if (loop)
            out = control(out, token_rows(messages, 3), p + "loop_ffn");
        out = ffn(norm(out, weight(p + "ffn_norm.weight", {width}), eps), p, layer);
        out = convolve(out,
                       weight(p + "ffn_post_conv.weight", {width, kernel}),
                       "conv_state_l" + std::to_string(slot) + "_ffn");
        return add(residual, out);
    }

    GgufMetadata meta;
    std::string arch;
    GraphEmitter e;
    int64_t layers = 0, width = 0, kernel = 0, block_size = 0, relative = 0, extent = 0, window = 0;
    int64_t experts = 0, serial = 0;
    float eps = 0, qk_clip = 0, rel_clip = 0, gate_cap = 0, up_cap = 0;
    float route_scale = 1;
    std::vector<int64_t> sliding;
    Value tau, distance_batch, distance_rows_cols;
    Value relative_indices[2], half_masks[2];
};
}  // namespace

std::shared_ptr<ModelBuilder> make_inkling_builder(const BuildContext& context) {
    return std::make_shared<InklingBuilder>(context);
}
}  // namespace ov::frontend::gguf
