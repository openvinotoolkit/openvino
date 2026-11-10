// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bagel_infer_request.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/runtime/make_tensor.hpp"

namespace {

// fp16 lowest: keeps the NPU fp16 softmax away from -inf arithmetic
constexpr float kMaskedOut = -65504.0f;

ov::Shape zero_dynamic_dims(const ov::PartialShape& shape) {
    ov::Shape result;
    for (const auto& dim : shape) {
        result.push_back(dim.is_static() ? dim.get_length() : 0);
    }
    return result;
}

uint8_t* bytes(const ov::SoPtr<ov::ITensor>& tensor) {
    return static_cast<uint8_t*>(tensor->data());
}

// Copies `rows` leading rows of `src` into `dst`; both are row-major with equal row size
void copy_rows(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst, std::size_t rows) {
    OPENVINO_ASSERT(src->get_element_type() == dst->get_element_type(), "NPUW_BAGEL: element type mismatch");
    const auto row_bytes =
        src->get_shape().empty() || src->get_shape()[0] == 0 ? 0 : src->get_byte_size() / src->get_shape()[0];
    std::memcpy(bytes(dst), bytes(src), rows * row_bytes);
}

// Copies a [1, K, len, D] block between tensors with different lengths on axis 2
void copy_seq_block(const ov::SoPtr<ov::ITensor>& src,
                    std::size_t src_off,
                    const ov::SoPtr<ov::ITensor>& dst,
                    std::size_t dst_off,
                    std::size_t len) {
    const auto& ss = src->get_shape();
    const auto& ds = dst->get_shape();
    const auto heads = ss[1];
    const auto row_bytes = ss[3] * src->get_element_type().size();
    for (std::size_t h = 0; h < heads; ++h) {
        std::memcpy(bytes(dst) + (h * ds[2] + dst_off) * row_bytes,
                    bytes(src) + (h * ss[2] + src_off) * row_bytes,
                    len * row_bytes);
    }
}

}  // namespace

ov::npuw::BagelInferRequest::BagelInferRequest(std::shared_ptr<const BagelCompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(std::move(compiled_model)),
      m_pending(m_compiled_model->outputs().size(), false) {
    const auto& inputs = m_compiled_model->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        for (const auto& name : inputs[i].get_names()) {
            m_input_idx.emplace(name, i);
        }
    }
    for (const auto& ports : {m_compiled_model->inputs(), m_compiled_model->outputs()}) {
        for (const auto& port : ports) {
            ov::ISyncInferRequest::set_tensor(
                port,
                ov::make_tensor(port.get_element_type(), zero_dynamic_dims(port.get_partial_shape())));
        }
    }
}

ov::npuw::BagelInferRequest::Variant& ov::npuw::BagelInferRequest::variant_for(
    const ov::SoPtr<ov::ICompiledModel>& compiled) {
    auto it = m_variants.find(compiled._ptr.get());
    if (it != m_variants.end()) {
        return it->second;
    }
    Variant variant;
    variant.request = compiled->create_infer_request();
    // Variants are reshaped clones of the same model, so input ports keep their order
    OPENVINO_ASSERT(compiled->inputs().size() == m_compiled_model->inputs().size(),
                    "NPUW_BAGEL: static variant inputs differ from the model inputs");
    variant.inputs = compiled->inputs();
    const auto& inner = compiled->outputs();
    for (std::size_t i = 0; i < m_compiled_model->outputs().size(); ++i) {
        if (inner.size() == m_compiled_model->outputs().size()) {
            variant.outputs.push_back(inner[i]);
            continue;
        }
        // Some outputs were dropped from the variant, so the rest is matched by name
        const auto& name = m_compiled_model->outputs()[i].get_any_name();
        const auto it = std::find_if(inner.begin(), inner.end(), [&](const ov::Output<const ov::Node>& p) {
            return p.get_names().count(name) > 0;
        });
        variant.outputs.push_back(it == inner.end() ? std::nullopt : std::optional(*it));
    }
    return m_variants.emplace(compiled._ptr.get(), std::move(variant)).first->second;
}

ov::SoPtr<ov::ITensor> ov::npuw::BagelInferRequest::input(const std::string& name) const {
    const auto it = m_input_idx.find(name);
    OPENVINO_ASSERT(it != m_input_idx.end(), "NPUW_BAGEL: model has no input ", name);
    return ov::ISyncInferRequest::get_tensor(m_compiled_model->inputs()[it->second]);
}

ov::SoPtr<ov::ITensor> ov::npuw::BagelInferRequest::inner_input(const Variant& variant, const std::string& name) const {
    return variant.request->get_tensor(variant.inputs[m_input_idx.at(name)]);
}

void ov::npuw::BagelInferRequest::infer() {
    if (m_compiled_model->mode() == BagelCompiledModel::Mode::LANGUAGE_MODEL) {
        infer_language_model();
    } else {
        infer_per_shape();
    }
}

void ov::npuw::BagelInferRequest::infer_per_shape() {
    BagelCompiledModel::ShapeKey key;
    for (const auto& port : m_compiled_model->inputs()) {
        key.push_back(ov::ISyncInferRequest::get_tensor(port)->get_shape());
    }
    auto& variant = variant_for(m_compiled_model->shape_variant(key));
    const auto& inputs = m_compiled_model->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        variant.request->set_tensor(variant.inputs[i], ov::ISyncInferRequest::get_tensor(inputs[i]));
    }
    variant.request->infer();
    const auto& outputs = m_compiled_model->outputs();
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        ov::ISyncInferRequest::set_tensor(outputs[i], variant.request->get_tensor(*variant.outputs[i]));
    }
}

void ov::npuw::BagelInferRequest::infer_language_model() {
    const auto embeds = input("inputs_embeds");
    const auto und = input("und_index");
    const auto gen = input("gen_index");
    const auto mask = input("attention_mask");
    const auto past0 = input("past_key_values.0.key");

    LMCall call;
    call.seq = embeds->get_shape()[0];
    call.kv = past0->get_shape()[2];
    call.und = und->get_size();
    call.gen = gen->get_size();
    OPENVINO_ASSERT(call.und + call.gen == call.seq,
                    "NPUW_BAGEL: und_index and gen_index must cover inputs_embeds, got ",
                    call.und,
                    " + ",
                    call.gen,
                    " != ",
                    call.seq);
    OPENVINO_ASSERT(mask->get_element_type() == ov::element::f32, "NPUW_BAGEL: attention_mask must be f32");

    // check_tensors() is skipped for the dynamic ports, so every size the copies below rely on is checked here
    OPENVINO_ASSERT(call.seq > 0, "NPUW_BAGEL: inputs_embeds must not be empty");
    OPENVINO_ASSERT(input("position_ids")->get_size() == call.seq,
                    "NPUW_BAGEL: position_ids must have ",
                    call.seq,
                    " elements");
    OPENVINO_ASSERT(mask->get_shape() == ov::Shape({1, 1, call.seq, call.kv + call.seq}),
                    "NPUW_BAGEL: attention_mask must be [1, 1, ",
                    call.seq,
                    ", ",
                    call.kv + call.seq,
                    "], got ",
                    mask->get_shape());
    if (call.gen > 0) {
        OPENVINO_ASSERT(input("x_t")->get_shape()[0] == call.gen && input("vae_position_ids")->get_size() == call.gen,
                        "NPUW_BAGEL: x_t and vae_position_ids must have one row per gen_index entry");
    }
    for (const auto& item : m_input_idx) {
        if (item.first.rfind("past_key_values.", 0) == 0) {
            OPENVINO_ASSERT(input(item.first)->get_shape()[2] == call.kv,
                            "NPUW_BAGEL: all past_key_values inputs must have length ",
                            call.kv);
        }
    }
    // The indices drive scatters on the device, so they must address rows of inputs_embeds
    for (const auto& indices : {und, gen}) {
        const auto* begin = indices->data<const int64_t>();
        OPENVINO_ASSERT(std::all_of(begin,
                                    begin + indices->get_size(),
                                    [&](int64_t i) {
                                        return i >= 0 && static_cast<std::size_t>(i) < call.seq;
                                    }),
                        "NPUW_BAGEL: und_index and gen_index values must be in [0, ",
                        call.seq,
                        ")");
    }

    // The NPU can't take zero-size tensors: a call without generation tokens gets one masked
    // dummy generation token in the padding
    const bool dummy = call.gen == 0;
    const std::size_t n_gen = dummy ? 1 : call.gen;
    call.seq_padded = dummy ? m_compiled_model->seq_bucket(call.seq + 1) : call.seq;
    call.kv_padded = m_compiled_model->kv_bucket(std::max<std::size_t>(call.kv, 1));
    const auto S = call.seq, Sb = call.seq_padded, P = call.kv, Pb = call.kv_padded;

    auto& variant = variant_for(m_compiled_model->lm_variant({Sb, n_gen, Pb}));

    const auto zero = [](const ov::SoPtr<ov::ITensor>& t) {
        std::memset(t->data(), 0, t->get_byte_size());
    };

    auto in_embeds = inner_input(variant, "inputs_embeds");
    zero(in_embeds);
    copy_rows(embeds, in_embeds, S);

    auto in_pos = inner_input(variant, "position_ids");
    copy_rows(input("position_ids"), in_pos, S);

    copy_rows(input("timestep"), inner_input(variant, "timestep"), 1);

    // Padded KV slots are masked out for every row and real rows never see padded rows. Padded rows
    // repeat the last real row (same position and context) plus themselves: a row that only sees
    // itself acts as an attention sink, whose outlier activations overflow fp16 and turn into NaN
    auto in_mask = inner_input(variant, "attention_mask");
    {
        auto* pos = in_pos->data<int64_t>();
        std::fill(pos + S, pos + Sb, pos[S - 1]);

        const auto* src = mask->data<const float>();
        auto* dst = in_mask->data<float>();
        const auto src_cols = P + S, dst_cols = Pb + Sb;
        std::fill_n(dst, Sb * dst_cols, kMaskedOut);
        for (std::size_t r = 0; r < S; ++r) {
            const auto* s = src + r * src_cols;
            auto* d = dst + r * dst_cols;
            std::transform(s, s + P, d, [](float v) {
                return std::max(v, kMaskedOut);
            });
            std::transform(s + P, s + P + S, d + Pb, [](float v) {
                return std::max(v, kMaskedOut);
            });
        }
        for (std::size_t r = S; r < Sb; ++r) {
            std::copy_n(dst + (S - 1) * dst_cols, dst_cols, dst + r * dst_cols);
            dst[r * dst_cols + Pb + r] = 0.0f;
        }
    }

    auto in_und = inner_input(variant, "und_index");
    auto in_gen = inner_input(variant, "gen_index");
    auto in_x_t = inner_input(variant, "x_t");
    auto in_vae_pos = inner_input(variant, "vae_position_ids");
    if (dummy) {
        // Real tokens and all padded rows but the last are "understanding"; the last is the dummy
        auto* und_dst = in_und->data<int64_t>();
        std::copy_n(und->data<const int64_t>(), call.und, und_dst);
        for (std::size_t r = S; r + 1 < Sb; ++r) {
            und_dst[call.und + (r - S)] = static_cast<int64_t>(r);
        }
        in_gen->data<int64_t>()[0] = static_cast<int64_t>(Sb - 1);
        zero(in_x_t);
        zero(in_vae_pos);
    } else {
        copy_rows(und, in_und, call.und);
        copy_rows(gen, in_gen, call.gen);
        copy_rows(input("x_t"), in_x_t, call.gen);
        copy_rows(input("vae_position_ids"), in_vae_pos, call.gen);
    }

    for (const auto& item : m_input_idx) {
        const auto& name = item.first;
        if (name.rfind("past_key_values.", 0) != 0) {
            continue;
        }
        auto in_past = inner_input(variant, name);
        zero(in_past);
        if (P > 0) {
            copy_seq_block(input(name), 0, in_past, 0, P);
        }
    }

    variant.request->infer();

    m_last = &variant;
    m_call = call;
    std::fill(m_pending.begin(), m_pending.end(), true);
}

void ov::npuw::BagelInferRequest::crop_output(std::size_t idx) {
    const auto& port = m_compiled_model->outputs()[idx];
    const auto& name = port.get_any_name();
    const auto& c = m_call;
    m_pending[idx] = false;
    if (!m_last->outputs[idx].has_value()) {
        // Not computed by this variant (logits outside decode): reported with zero rows
        auto shape = port.get_partial_shape();
        shape[0] = 0;
        ov::ISyncInferRequest::set_tensor(port, ov::make_tensor(port.get_element_type(), shape.to_shape()));
        return;
    }
    const auto src = m_last->request->get_tensor(*m_last->outputs[idx]);

    ov::Shape shape = src->get_shape();
    if (name == "logits") {
        shape[0] = c.und;
    } else if (name == "v_t") {
        shape[0] = c.gen;
    } else if (name.rfind("present.", 0) == 0) {
        shape[2] = c.kv + c.seq;
    } else {
        OPENVINO_THROW("NPUW_BAGEL: no crop rule for output ", name);
    }

    auto dst = ov::make_tensor(src->get_element_type(), shape);
    if (name.rfind("present.", 0) == 0) {
        copy_seq_block(src, 0, dst, 0, c.kv);
        copy_seq_block(src, c.kv_padded, dst, c.kv, c.seq);
    } else {
        copy_rows(src, dst, shape[0]);
    }
    ov::ISyncInferRequest::set_tensor(port, dst);
}

ov::SoPtr<ov::ITensor> ov::npuw::BagelInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    const auto& outputs = m_compiled_model->outputs();
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (m_pending[i] && outputs[i] == port) {
            const_cast<BagelInferRequest*>(this)->crop_output(i);
            break;
        }
    }
    return ov::ISyncInferRequest::get_tensor(port);
}
