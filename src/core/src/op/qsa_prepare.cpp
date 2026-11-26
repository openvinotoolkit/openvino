// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_prepare.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "itt.hpp"
#include "openvino/core/type/float16.hpp"
#include "qsa_prepare_shape_inference.hpp"

namespace ov::op::v17 {

QSAPrepare::QSAPrepare(const OutputVector& inputs, int64_t key_block_size, float epsilon, const std::string& pooling)
    : Op(inputs),
      m_key_block_size(key_block_size),
      m_epsilon(epsilon),
      m_pooling(pooling) {
    constructor_validate_and_infer_types();
}

QSAPrepare::QSAPrepare(const Output<Node>& raw_index_keys,
                       const Output<Node>& past_block_index_keys,
                       const Output<Node>& key_length,
                       const Output<Node>& rope_cos,
                       const Output<Node>& rope_sin,
                       const Output<Node>& k_norm_weight,
                       const Output<Node>& past_incomplete_raw_keys,
                       const Output<Node>& past_incomplete_rope,
                       int64_t key_block_size,
                       float epsilon)
    : QSAPrepare(OutputVector{raw_index_keys,
                              past_block_index_keys,
                              key_length,
                              rope_cos,
                              rope_sin,
                              k_norm_weight,
                              past_incomplete_raw_keys,
                              past_incomplete_rope},
                 key_block_size,
                 epsilon) {}

void QSAPrepare::validate_and_infer_types() {
    OV_OP_SCOPE(v17_QSAPrepare_validate_and_infer_types);
    const auto input_count = get_input_size();
    NODE_VALIDATION_CHECK(this,
                          m_pooling == "mean" || m_pooling == "softmax_gate",
                          "The attribute pooling must be \"mean\" or \"softmax_gate\". Got: ",
                          m_pooling);
    const size_t expected_inputs = is_softmax_gate_pooling() ? 9 : 8;
    NODE_VALIDATION_CHECK(this,
                          input_count == expected_inputs,
                          "The number of inputs must be ",
                          expected_inputs,
                          " for pooling ",
                          m_pooling,
                          ". Got: ",
                          input_count);
    NODE_VALIDATION_CHECK(this,
                          m_epsilon > 0.0f,
                          "The attribute epsilon must be a positive floating point number. Got: ",
                          m_epsilon);

    static constexpr const char* data_input_names[] = {"raw_index_keys",
                                                       "past_block_index_keys",
                                                       nullptr,
                                                       "rope_cos",
                                                       "rope_sin",
                                                       "k_norm_weight",
                                                       "past_incomplete_raw_keys",
                                                       "past_incomplete_rope",
                                                       "pool_bias"};
    element::Type data_et = get_input_element_type(0);
    for (size_t i = 1; i < input_count; ++i) {
        if (data_input_names[i] == nullptr) {
            continue;
        }
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(data_et, data_et, get_input_element_type(i)),
                              "The element type of ",
                              data_input_names[i],
                              " must be the same as of raw_index_keys. Got: ",
                              get_input_element_type(i));
    }
    NODE_VALIDATION_CHECK(this,
                          data_et.is_dynamic() || data_et.is_real(),
                          "The element type of the index key inputs must be a floating point type. Got: ",
                          data_et);

    const auto& key_length_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          key_length_et.is_dynamic() || key_length_et == element::i32 || key_length_et == element::i64,
                          "The element type of key_length must be i32 or i64. Got: ",
                          key_length_et);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, data_et, output_shapes[0]);
    set_output_type(1, data_et, output_shapes[1]);
    set_output_type(2, data_et, output_shapes[2]);
}

bool QSAPrepare::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v17_QSAPrepare_visit_attributes);
    visitor.on_attribute("key_block_size", m_key_block_size);
    visitor.on_attribute("epsilon", m_epsilon);
    visitor.on_attribute("pooling", m_pooling);
    return true;
}

std::shared_ptr<Node> QSAPrepare::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v17_QSAPrepare_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<QSAPrepare>(new_args, m_key_block_size, m_epsilon, m_pooling);
}

// Reference: no RoPE (the rotation is not implemented here; has_evaluate() is false when it is enabled).
bool QSAPrepare::has_evaluate() const {
    const auto et = get_input_element_type(0);
    const bool rope_off = ov::util::is_empty_constant_tensor(input_value(3));
    return rope_off && (et == element::f32 || et == element::f16);
}

bool QSAPrepare::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v17_QSAPrepare_evaluate);
    if (!has_evaluate())
        return false;
    const bool gate = is_softmax_gate_pooling();
    const auto& rs = inputs[0].get_shape();  // [N, L, W]
    const size_t N = rs[0], L = rs[1], W = rs[2], Di = gate ? W / 2 : W;
    const size_t R = static_cast<size_t>(m_key_block_size);
    const size_t Bp = inputs[1].get_shape()[1];
    std::vector<int64_t> kl(inputs[2].get_size());
    for (size_t i = 0; i < kl.size(); i++)
        kl[i] = inputs[2].get_element_type() == element::i32 ? inputs[2].data<const int32_t>()[i]
                                                             : inputs[2].data<const int64_t>()[i];
    auto K_of = [&](size_t n) {
        return kl[kl.size() == 1 ? 0 : n];
    };
    size_t max_blocks = 0;
    for (size_t n = 0; n < N; n++)
        max_blocks = std::max(max_blocks, static_cast<size_t>(K_of(n) / static_cast<int64_t>(R)));
    outputs[0].set_shape({N, max_blocks, Di});
    outputs[1].set_shape({N, R - 1, W});
    outputs[2].set_shape({N, 0});
    const bool knorm = !gate && inputs[5].get_size() > 0;
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* raw = inputs[0].data<const T>();
        const T* past_blocks = inputs[1].data<const T>();
        const T* past_inc = inputs[6].data<const T>();
        const T* bias = gate ? inputs[8].data<const T>() : nullptr;  // [Di, R]
        T* blocks = outputs[0].data<T>();
        T* inc_out = outputs[1].data<T>();
        std::fill(blocks, blocks + outputs[0].get_size(), T(0));
        std::fill(inc_out, inc_out + outputs[1].get_size(), T(0));
        for (size_t n = 0; n < N; n++) {
            const int64_t K = K_of(n), P = K - static_cast<int64_t>(L);
            const int64_t past_complete = P > 0 ? P / static_cast<int64_t>(R) : 0;
            const int64_t start = past_complete * static_cast<int64_t>(R);  // first token of the incomplete block
            // token stream from `start`: the past incomplete rows, then the new rows
            auto row = [&](int64_t pos) -> const T* {
                return pos < P ? past_inc + (n * (R - 1) + static_cast<size_t>(pos - start)) * W
                               : raw + (n * L + static_cast<size_t>(pos - P)) * W;
            };
            for (int64_t b = 0; b < past_complete && b < static_cast<int64_t>(Bp); b++)
                std::copy_n(past_blocks + (n * Bp + b) * Di, Di, blocks + (n * max_blocks + b) * Di);
            const int64_t total = K / static_cast<int64_t>(R);
            for (int64_t b = past_complete; b < total; b++) {
                T* dst = blocks + (n * max_blocks + b) * Di;
                std::vector<float> acc(Di, 0.0f);
                for (size_t c = 0; c < Di; c++) {
                    if (gate) {
                        float mx = -INFINITY;
                        for (size_t s = 0; s < R; s++)
                            mx = std::max(
                                mx,
                                static_cast<float>(row(b * R + s)[Di + c]) + static_cast<float>(bias[c * R + s]));
                        float num = 0.0f, den = 0.0f;
                        for (size_t s = 0; s < R; s++) {
                            const T* r = row(b * R + s);
                            const float e =
                                std::exp(static_cast<float>(r[Di + c]) + static_cast<float>(bias[c * R + s]) - mx);
                            num += e * static_cast<float>(r[c]);
                            den += e;
                        }
                        acc[c] = num / den;
                    } else {
                        for (size_t s = 0; s < R; s++)
                            acc[c] += static_cast<float>(row(b * R + s)[c]);
                        acc[c] /= static_cast<float>(R);
                    }
                }
                if (knorm) {
                    float ms = 0.0f;
                    for (float x : acc)
                        ms += x * x;
                    const float inv = 1.0f / std::sqrt(ms / static_cast<float>(Di) + m_epsilon);
                    for (size_t c = 0; c < Di; c++)
                        acc[c] *= inv * static_cast<float>(inputs[5].data<const T>()[c]);
                }
                for (size_t c = 0; c < Di; c++)
                    dst[c] = static_cast<T>(acc[c]);
            }
            // the new incomplete block: tokens [total * R, K)
            for (int64_t pos = total * static_cast<int64_t>(R); pos < K; pos++)
                std::copy_n(row(pos),
                            W,
                            inc_out + (n * (R - 1) + static_cast<size_t>(pos - total * static_cast<int64_t>(R))) * W);
        }
    };
    if (inputs[0].get_element_type() == element::f32)
        run(float{});
    else
        run(ov::float16{});
    return true;
}

}  // namespace ov::op::v17
