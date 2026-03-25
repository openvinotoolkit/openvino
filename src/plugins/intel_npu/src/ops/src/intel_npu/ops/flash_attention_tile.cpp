// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <intel_npu/ops/flash_attention_tile.hpp>
#include <limits>
#include <numeric>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

using namespace std;

namespace {

void flash_attention_evaluate(const float* query,
                              const float* key,
                              const float* value,
                              float* running_output,
                              float* running_max,
                              float* running_sum,
                              const float* attention_mask,
                              bool is_head,
                              bool is_tail,
                              int32_t mask_batch_stride,
                              int32_t mask_head_stride,
                              int32_t B,
                              int32_t H_q,
                              int32_t H_kv,
                              int32_t L,
                              int32_t S,
                              int32_t E,
                              int32_t Ev) {
    std::vector<float> scores(S);
    std::vector<float> exp_scores(S);

    const auto gqa_group_size = H_q / H_kv;

    for (int32_t b = 0; b < B; ++b) {
        for (int32_t h = 0; h < H_q; ++h) {
            auto kv_h = h / gqa_group_size;
            auto q_ptr = query + b * H_q * L * E + h * L * E;
            auto k_ptr = key + b * H_kv * S * E + kv_h * S * E;
            auto v_ptr = value + b * H_kv * S * Ev + kv_h * S * Ev;

            const float* m_ptr = nullptr;
            if (attention_mask != nullptr) {
                m_ptr = attention_mask + b * mask_batch_stride + h * mask_head_stride;
            }

            auto out_ptr = running_output + b * H_q * L * Ev + h * L * Ev;
            auto max_ptr = running_max + b * H_q * L + h * L;
            auto sum_ptr = running_sum + b * H_q * L + h * L;

            for (int32_t l = 0; l < L; ++l) {
                auto q_row = q_ptr + l * E;
                auto m_row = m_ptr ? (m_ptr + l * S) : nullptr;
                auto o_row = out_ptr + l * Ev;

                auto tile_max = -std::numeric_limits<float>::infinity();
                for (int32_t s = 0; s < S; ++s) {
                    auto k_row = k_ptr + s * E;

                    auto score = 0.f;
                    for (int32_t e = 0; e < E; ++e) {
                        score += q_row[e] * k_row[e];
                    }

                    if (m_row) {
                        score += m_row[s];
                    }

                    scores[s] = score;
                    tile_max = std::max(tile_max, score);
                }

                for (int32_t s = 0; s < S; ++s) {
                    exp_scores[s] = std::exp(scores[s] - tile_max);
                }

                auto tile_sum = 0.f;
                for (int32_t s = 0; s < S; ++s) {
                    tile_sum += exp_scores[s];
                }

                auto old_max = max_ptr[l];
                auto new_max = std::max(old_max, tile_max);

                auto correction = std::exp(old_max - new_max);
                auto tile_scale = std::exp(tile_max - new_max);

                for (int32_t ev = 0; ev < Ev; ++ev) {
                    o_row[ev] = o_row[ev] * correction;
                }
                for (int32_t s = 0; s < S; ++s) {
                    exp_scores[s] *= tile_scale;
                }

                max_ptr[l] = new_max;
                sum_ptr[l] = sum_ptr[l] * correction + tile_sum * tile_scale;

                for (int32_t s = 0; s < S; ++s) {
                    auto v_row = v_ptr + s * Ev;
                    auto w = exp_scores[s];
                    for (int32_t ev = 0; ev < Ev; ++ev) {
                        auto acc = o_row[ev] + w * v_row[ev];
                        o_row[ev] = acc;
                    }
                }

                if (is_tail) {
                    auto inv_sum = 1.f / sum_ptr[l];
                    for (int32_t ev = 0; ev < Ev; ++ev) {
                        o_row[ev] = o_row[ev] * inv_sum;
                    }
                }
            }
        }
    }
}

}  // namespace

namespace ov::intel_npu::op {

using namespace ov::op;

static std::vector<ov::PartialShape> shape_infer(const FlashAttentionTile* op,
                                                 const std::vector<ov::PartialShape>& input_shapes) {
    using DimType = typename ov::PartialShape::value_type;
    const auto& inputs_count = input_shapes.size();
    const auto& has_attention_mask = (inputs_count >= 7) && (input_shapes[6].size() > 1);
    NODE_VALIDATION_CHECK(op, inputs_count == 6 || has_attention_mask);

    DimType e_dim{};
    DimType l_dim{};
    DimType s_dim{};
    DimType ev_dim{};

    const auto shape_has_static_rank = [](const PartialShape& shape) {
        return shape.rank().is_static();
    };
    const auto inputs_have_static_rank = std::all_of(input_shapes.begin(), input_shapes.end(), shape_has_static_rank);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, inputs_have_static_rank, "Inputs with dynamic rank are not supported.");

    const auto& query_shape = input_shapes[0];
    const auto& query_rank = query_shape.rank();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           query_rank.get_length() >= 3,
                           "Query input rank length must be at least 3 or more.");

    l_dim = *(query_shape.end() - 2);
    e_dim = *(query_shape.end() - 1);

    // Split query dims into true batch dims and head dim.
    // For rank 3: [H, L, E] -> batch_dims=[], head_dim=H
    // For rank 4: [B, H, L, E] -> batch_dims=[B], head_dim=H
    auto q_head_dim = *(query_shape.end() - 3);
    auto query_batch_dims = query_shape;
    query_batch_dims.resize(query_shape.size() - 3);

    // Helper: check that KV head dim is compatible with Q head dim (GQA: Q heads divisible by KV heads)
    auto is_gqa_compatible = [](const DimType& q_head, const DimType& kv_head) -> bool {
        if (q_head.is_static() && kv_head.is_static()) {
            return kv_head.get_length() != 0 && q_head.get_length() % kv_head.get_length() == 0;
        }
        return true;  // allow dynamic dims
    };

    const auto& key_shape = input_shapes[1];
    const auto& key_rank = key_shape.rank();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           key_rank.get_length() >= 3,
                           "Key input rank length must be at least 3 or more.");
    auto key_batch_dims = ov::PartialShape(std::vector<DimType>(key_shape.begin(), key_shape.end() - 3));
    auto kv_head_dim = *(key_shape.end() - 3);
    const bool key_input_correctness =
        ov::PartialShape::merge_into(query_batch_dims, key_batch_dims) &&
        is_gqa_compatible(q_head_dim, kv_head_dim) && DimType::merge(e_dim, e_dim, *(key_shape.end() - 1));
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           key_input_correctness,
                           "Key input shape not compatible with other inputs.");

    s_dim = *(key_shape.end() - 2);

    const auto& value_shape = input_shapes[2];
    const auto& value_rank = value_shape.rank();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           value_rank.get_length() >= 3,
                           "Value input rank length must be at least 3 or more.");
    auto value_batch_dims = ov::PartialShape(std::vector<DimType>(value_shape.begin(), value_shape.end() - 3));
    auto v_head_dim = *(value_shape.end() - 3);
    const bool value_input_correctness =
        ov::PartialShape::merge_into(query_batch_dims, value_batch_dims) &&
        is_gqa_compatible(q_head_dim, v_head_dim) && DimType::merge(kv_head_dim, kv_head_dim, v_head_dim) &&
        DimType::merge(s_dim, s_dim, *(value_shape.end() - 2));
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           value_input_correctness,
                           "Value input shape not compatible with other inputs.");

    ev_dim = *(value_shape.end() - 1);

    // Reconstruct full Q batch dims (including head dim) for running state validation
    auto q_full_batch_dims = query_batch_dims;
    q_full_batch_dims.push_back(q_head_dim);

    const auto& running_output_shape = input_shapes[3];
    const auto& running_output_rank = running_output_shape.rank();
    const bool running_output_correctness =
        running_output_rank.get_length() >= 3 &&
        ov::PartialShape::broadcast_merge_into(
            q_full_batch_dims,
            ov::PartialShape(std::vector<DimType>(running_output_shape.begin(), running_output_shape.end() - 2)),
            AutoBroadcastType::NUMPY) &&
        (*(running_output_shape.end() - 1) == ev_dim) && (*(running_output_shape.end() - 2) == l_dim);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           running_output_correctness,
                           "Running Output input shape not compatible with other inputs.");

    const auto& running_max_shape = input_shapes[4];
    const auto& running_max_rank = running_max_shape.rank();
    const bool running_max_correctness =
        running_max_rank.get_length() >= 2 &&
        ov::PartialShape::broadcast_merge_into(
            q_full_batch_dims,
            ov::PartialShape(std::vector<DimType>(running_max_shape.begin(), running_max_shape.end() - 1)),
            AutoBroadcastType::NUMPY) &&
        (*(running_max_shape.end() - 1) == l_dim);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           running_max_correctness,
                           "Running Max input shape not compatible with other inputs.");

    const auto& running_sum_shape = input_shapes[5];
    const auto& running_sum_rank = running_sum_shape.rank();
    const bool running_sum_correctness =
        running_sum_rank.get_length() >= 2 &&
        ov::PartialShape::broadcast_merge_into(
            q_full_batch_dims,
            ov::PartialShape(std::vector<DimType>(running_sum_shape.begin(), running_sum_shape.end() - 1)),
            AutoBroadcastType::NUMPY) &&
        (*(running_sum_shape.end() - 1) == l_dim);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           running_sum_correctness,
                           "Running Sum input shape not compatible with other inputs.");

    if (has_attention_mask) {
        const auto& attention_mask = input_shapes[6];
        const auto& attention_mask_rank = attention_mask.rank();
        const auto& attention_mask_rank_len = attention_mask_rank.get_length();
        bool attention_mask_input_correctness = attention_mask_rank_len >= 2 &&
                                                DimType::broadcast_merge(l_dim, l_dim, *(attention_mask.end() - 2)) &&
                                                DimType::broadcast_merge(s_dim, s_dim, *(attention_mask.end() - 1));
        if (attention_mask_rank_len >= 3) {
            attention_mask_input_correctness =
                attention_mask_input_correctness &&
                ov::PartialShape::broadcast_merge_into(
                    q_full_batch_dims,
                    ov::PartialShape(std::vector<DimType>(attention_mask.begin(), attention_mask.end() - 2)),
                    AutoBroadcastType::NUMPY);
        }
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               attention_mask_input_correctness,
                               "Attention mask input shape not compatible with other inputs.");
    }

    auto result_running_output_shape = q_full_batch_dims;
    result_running_output_shape.push_back(l_dim);
    result_running_output_shape.push_back(ev_dim);

    // Running max and sum have the same shape
    auto result_running_max_and_sum_shape = q_full_batch_dims;
    result_running_max_and_sum_shape.push_back(l_dim);

    return {result_running_output_shape, result_running_max_and_sum_shape, result_running_max_and_sum_shape};
}

FlashAttentionTile::FlashAttentionTile(const OutputVector& inputs, Config config)
    : op::Op(inputs),
      m_config(std::move(config)) {
    constructor_validate_and_infer_types();
}

FlashAttentionTile::FlashAttentionTile(const Output<Node>& query,
                                       const Output<Node>& key,
                                       const Output<Node>& value,
                                       const Output<Node>& running_output,
                                       const Output<Node>& running_max,
                                       const Output<Node>& running_sum,
                                       Config config)
    : FlashAttentionTile({query, key, value, running_output, running_max, running_sum}, std::move(config)) {}

FlashAttentionTile::FlashAttentionTile(const Output<Node>& query,
                                       const Output<Node>& key,
                                       const Output<Node>& value,
                                       const Output<Node>& running_output,
                                       const Output<Node>& running_max,
                                       const Output<Node>& running_sum,
                                       const Output<Node>& attn_mask,
                                       Config config)
    : FlashAttentionTile({query, key, value, running_output, running_max, running_sum, attn_mask}, std::move(config)) {}

void FlashAttentionTile::validate_and_infer_types() {
    const auto attention_mask_idx = 6;

    auto query_element_type = get_input_element_type(0);
    const auto input_size = static_cast<int32_t>(get_input_size());

    if (input_size >= 7) {
        const auto& attention_type = get_input_element_type(attention_mask_idx);
        NODE_VALIDATION_CHECK(
            this,
            attention_type.is_real() || attention_type == element::boolean || attention_type.is_dynamic(),
            "The element type of attention_mask must be either floating-point or boolean.");
    }

    // Support only f32 reference
    for (int32_t i = 1; i < input_size; i++) {
        const auto& element_type = get_input_element_type(i);

        if (i == attention_mask_idx && (element_type == element::boolean)) {
            // Skip checking attention_mask in loop when boolean or skipped to not affect merged dtype.
            continue;
        }

        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(query_element_type, query_element_type, element_type),
                              "Mixed input types are not supported.");
    }

    NODE_VALIDATION_CHECK(this,
                          query_element_type.is_real() || query_element_type.is_dynamic(),
                          "The element type of the input tensor must be a floating-point.");

    const auto& input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    // Have the same output element types as the input running tensors
    const auto outputs_element_type =
        std::vector<ov::element::Type>{get_input_element_type(3), get_input_element_type(4), get_input_element_type(5)};

    for (size_t i = 0; i < output_shapes.size(); i++) {
        set_output_type(i, outputs_element_type[i], output_shapes[i]);
    }
}

std::shared_ptr<Node> FlashAttentionTile::clone_with_new_inputs(const OutputVector& new_args) const {
    return std::make_shared<FlashAttentionTile>(new_args, m_config);
}

bool FlashAttentionTile::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("is_head", m_config.is_head);
    visitor.on_attribute("is_tail", m_config.is_tail);
    return true;
}

enum FlashAttentionInputs {
    QUERY = 0,           // [B, H, L, E]
    KEY = 1,             // [B, H, S, E]
    VALUE = 2,           // [B, H, S, Ev]
    RUNNING_OUTPUT = 3,  // [B, H, L, Ev]
    RUNNING_MAX = 4,     // [B, H, L]
    RUNNING_SUM = 5,     // [B, H, L]
    ATTENTION_MASK = 6,  // [B, H, L, S]
};

enum FlashAttentionOutputs {
    OUTPUT = 0,      // [B, H, L, Ev]
    OUTPUT_MAX = 1,  // [B, H, L]
    OUTPUT_SUM = 2   // [B, H, L]
};

bool FlashAttentionTile::has_evaluate() const {
    switch (get_input_element_type(0)) {
    case element::f16:
    case element::f32:
        return true;
    default:
        return false;
    }
};

template <typename T>
static void convert_to_float(const T* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
}

template <typename T>
static void convert_from_float(const float* src, T* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<T>(src[i]);
    }
}

static ov::Tensor to_f32(const ov::Tensor& tensor) {
    if (tensor.get_element_type() == ov::element::f32) {
        return tensor;
    }
    ov::Tensor result(ov::element::f32, tensor.get_shape());
    if (tensor.get_element_type() == ov::element::f16) {
        convert_to_float(tensor.data<const ov::float16>(), result.data<float>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::boolean) {
        // Additive mask: true (attend) -> 0.0, false (mask out) -> -inf
        const auto* src = tensor.data<const char>();
        auto* dst = result.data<float>();
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            dst[i] = src[i] ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    } else {
        OPENVINO_THROW("Unsupported element type in to_f32: ", tensor.get_element_type());
    }
    return result;
}

static void from_f32(const ov::Tensor& f32_tensor, ov::Tensor& dst_tensor) {
    if (dst_tensor.get_element_type() == ov::element::f32) {
        return;
    }
    if (dst_tensor.get_element_type() != ov::element::f16) {
        OPENVINO_THROW("Unsupported element type in from_f32: ", dst_tensor.get_element_type());
    }
    convert_from_float(f32_tensor.data<const float>(), dst_tensor.data<ov::float16>(), dst_tensor.get_size());
}

static bool evaluate_flash_attention_impl(ov::TensorVector& outputs,
                                          const ov::TensorVector& inputs,
                                          const FlashAttentionTile::Config& config) {
    const auto& query_shape = inputs[QUERY].get_shape();
    const auto B = static_cast<int32_t>(
        std::accumulate(query_shape.begin(), query_shape.end() - 3, size_t{1}, std::multiplies<size_t>()));
    const auto H_q = static_cast<int32_t>(*(query_shape.end() - 3));
    const auto L = static_cast<int32_t>(*(query_shape.end() - 2));
    const auto E = static_cast<int32_t>(*(query_shape.end() - 1));

    const auto& key_shape = inputs[KEY].get_shape();
    const auto H_kv = static_cast<int32_t>(*(key_shape.end() - 3));
    if (H_kv == 0 || H_q % H_kv != 0) {
        return false;
    }
    const auto S = static_cast<int32_t>(*(key_shape.end() - 2));

    const auto& value_shape = inputs[VALUE].get_shape();
    const auto Ev = static_cast<int32_t>(*(value_shape.end() - 1));

    const auto out_elems = B * H_q * L * Ev;
    const auto state_elems = B * H_q * L;

    const auto* query = inputs[QUERY].data<const float>();
    const auto* key = inputs[KEY].data<const float>();
    const auto* value = inputs[VALUE].data<const float>();

    const float* attention_mask = nullptr;
    int32_t mask_batch_stride = 0;
    int32_t mask_head_stride = 0;
    ov::Tensor mask_f32;
    if (inputs.size() >= 7 && inputs[ATTENTION_MASK].get_size() > 1) {
        mask_f32 = to_f32(inputs[ATTENTION_MASK]);
        attention_mask = mask_f32.data<const float>();
        const auto& mask_shape = inputs[ATTENTION_MASK].get_shape();
        const auto mask_rank = mask_shape.size();
        const auto mask_L = static_cast<int32_t>(*(mask_shape.end() - 2));
        const auto mask_S = static_cast<int32_t>(*(mask_shape.end() - 1));
        if (mask_rank >= 3) {
            const auto mask_H = static_cast<int32_t>(*(mask_shape.end() - 3));
            if (mask_H > 1) {
                mask_head_stride = mask_L * mask_S;
            }
            if (mask_rank >= 4) {
                int32_t mask_B = 1;
                for (size_t i = 0; i < mask_rank - 3; ++i) {
                    mask_B *= static_cast<int32_t>(mask_shape[i]);
                }
                if (mask_B > 1) {
                    mask_batch_stride = mask_H * mask_L * mask_S;
                }
            }
        }
    }

    auto* out_output = outputs[OUTPUT].data<float>();
    auto* out_max = outputs[OUTPUT_MAX].data<float>();
    auto* out_sum = outputs[OUTPUT_SUM].data<float>();

    const auto* in_output = inputs[RUNNING_OUTPUT].data<const float>();
    const auto* in_max = inputs[RUNNING_MAX].data<const float>();
    const auto* in_sum = inputs[RUNNING_SUM].data<const float>();

    std::copy_n(in_output, out_elems, out_output);
    std::copy_n(in_max, state_elems, out_max);
    std::copy_n(in_sum, state_elems, out_sum);

    // Update running state in-place
    flash_attention_evaluate(query,
                             key,
                             value,
                             out_output,
                             out_max,
                             out_sum,
                             attention_mask,
                             config.is_head,
                             config.is_tail,
                             mask_batch_stride,
                             mask_head_stride,
                             B,
                             H_q,
                             H_kv,
                             L,
                             S,
                             E,
                             Ev);
    return true;
}

inline bool FlashAttentionTile::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    auto input_type = get_input_element_type(QUERY);
    if (input_type == ov::element::f32) {
        return evaluate_flash_attention_impl(outputs, inputs, m_config);
    }

    if (input_type != ov::element::f16) {
        return false;
    }

    // Convert f16 inputs to f32 for computation
    ov::TensorVector f32_inputs;
    f32_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        f32_inputs.push_back(to_f32(inputs[i]));
    }

    // Allocate f32 outputs
    ov::TensorVector f32_outputs;
    f32_outputs.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        f32_outputs.emplace_back(ov::element::f32, outputs[i].get_shape());
    }

    if (!evaluate_flash_attention_impl(f32_outputs, f32_inputs, m_config)) {
        return false;
    }

    // Convert f32 outputs back to original types
    for (size_t i = 0; i < outputs.size(); ++i) {
        from_f32(f32_outputs[i], outputs[i]);
    }

    return true;
}

const FlashAttentionTile::Config& FlashAttentionTile::get_config() const {
    return m_config;
}

void FlashAttentionTile::set_config(Config config) {
    m_config = std::move(config);
}

}  // namespace ov::intel_npu::op
