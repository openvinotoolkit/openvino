// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_infer_request.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "npuw/infer_request_utils.hpp"
#include "npuw/logging.hpp"
#include "npuw/util.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace {
std::string safe_any_name(const ov::Output<const ov::Node>& port) {
    const auto& names = port.get_names();
    if (!names.empty()) {
        return port.get_any_name();
    }
    return {};
}

void throw_if_missing(const std::vector<std::string>& missing_ports) {
    if (!missing_ports.empty()) {
        std::stringstream ss;
        ss << "Can't match ports: ";
        for (std::size_t i = 0; i < missing_ports.size(); ++i) {
            if (i)
                ss << ", ";
            ss << (missing_ports[i].empty() ? "<unnamed>" : missing_ports[i]);
        }
        OPENVINO_THROW(ss.str());
    }
}

/**
 * @brief Gathers elements from the last dimension of a 3D tensor based on provided indices.
 *
 * This function extracts specific elements from the last dimension (L) of a source tensor
 * with shape [1, C, L] and places them into a destination buffer. The destination buffer
 * is treated as having a shape of [1, C, block].
 *
 * The operation effectively performs:
 * dst[0, c, t] = src[0, c, idx[t]]
 *
 *  Source (L=3)       Indices (idx)        Destination (block=5)
 * [ A | B | C ]   <-- [0, 0, 1, 2, 2] --> [ A | A | B | C | C ]
 */
template <typename T>
void gather_3d_last_dim(const ov::SoPtr<ov::ITensor>& src, T* dst, const std::vector<int64_t>& idx, std::size_t block) {
    // src: [1, C, L], dst: [1, C, block]
    const auto& shape = src->get_shape();
    OPENVINO_ASSERT(shape.size() == 3u);
    OPENVINO_ASSERT(shape[0] == 1u);
    const std::size_t channels = shape[1];
    const std::size_t phonemes_number = shape[2];

    OPENVINO_ASSERT(src->is_continuous());
    const T* src_p = src->data<const T>();

    // Zero fill destination buffer, as it might be not fully filled, remaining will be 0-padding.
    std::fill(dst, dst + (channels * block), T{});
    const std::size_t n = idx.size();

    for (std::size_t c = 0; c < channels; ++c) {
        const T* row = src_p + c * phonemes_number;
        T* out = dst + c * block;
        for (std::size_t t = 0; t < n; ++t) {
            const auto it = idx[t];
            OPENVINO_ASSERT(it >= 0);
            OPENVINO_ASSERT(static_cast<std::size_t>(it) < phonemes_number);
            out[t] = row[static_cast<std::size_t>(it)];
        }
    }
}

}  // namespace

void ov::npuw::KokoroInferRequest::init_tensor(const ov::Output<const ov::Node>& port) {
    ov::SoPtr<ITensor> tensor;
    tensor = ov::ISyncInferRequest::get_tensor(port);

    if (!tensor) {
        const auto& shape = port.get_partial_shape();
        const bool is_dynamic = shape.is_dynamic();
        ov::Shape tensor_shape;
        if (is_dynamic) {
            for (auto&& item : shape) {
                tensor_shape.push_back(item.is_static() ? item.get_length() : 0);
            }
        } else {
            tensor_shape = shape.to_shape();
        }

        tensor = ov::make_tensor(port.get_element_type(), tensor_shape);
        set_tensor(port, tensor);
    }
}

ov::npuw::KokoroInferRequest::KokoroInferRequest(const std::shared_ptr<ov::npuw::KokoroCompiledModel>& compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_kokoro_compiled_model(compiled_model) {
    const auto original_inputs = m_kokoro_compiled_model->inputs();

    for (const auto& input_port : original_inputs) {
        init_tensor(input_port);
    }
    for (const auto& output_port : m_kokoro_compiled_model->outputs()) {
        init_tensor(output_port);
    }

    OPENVINO_ASSERT(m_kokoro_compiled_model->model_a(), "Kokoro: Model A is not compiled");
    OPENVINO_ASSERT(m_kokoro_compiled_model->model_b(), "Kokoro: Model B is not compiled");
    m_model_a_request = m_kokoro_compiled_model->model_a()->create_infer_request();
    m_model_b_request = m_kokoro_compiled_model->model_b()->create_infer_request();

    // Map Model A inputs
    const auto a_inputs = m_model_a_request->get_compiled_model()->inputs();
    std::vector<std::string> missing_ports;
    for (const auto& a_in : a_inputs) {
        auto original_port = ov::npuw::util::find_port_by_names(original_inputs, a_in.get_names());
        if (original_port.has_value()) {
            m_model_a_in_map.push_back({a_in, original_port.value()});
        } else {
            missing_ports.push_back(safe_any_name(a_in));
        }
    }
    throw_if_missing(missing_ports);

    // Map Model A outputs
    const auto a_outputs = m_model_a_request->get_compiled_model()->outputs();
    auto pred_dur_port = ov::npuw::util::find_port_by_name(a_outputs, "pred_dur");
    auto en_left_port = ov::npuw::util::find_port_by_name(a_outputs, "en_left");
    auto asr_left_port = ov::npuw::util::find_port_by_name(a_outputs, "asr_left");
    OPENVINO_ASSERT(pred_dur_port.has_value(), "Kokoro Model A output 'pred_dur' not found");
    OPENVINO_ASSERT(en_left_port.has_value(), "Kokoro Model A output 'en_left' not found");
    OPENVINO_ASSERT(asr_left_port.has_value(), "Kokoro Model A output 'asr_left' not found");
    m_a_pred_dur = pred_dur_port.value();
    m_a_en_left = en_left_port.value();
    m_a_asr_left = asr_left_port.value();

    // Map Model B inputs
    const auto b_inputs = m_model_b_request->get_compiled_model()->inputs();
    std::vector<std::string> b_missing_ports;
    for (const auto& b_in : b_inputs) {
        const auto& names = b_in.get_names();

        if (std::count(names.begin(), names.end(), "en_block")) {
            m_b_en_block = b_in;
            continue;
        }
        if (std::count(names.begin(), names.end(), "asr_block")) {
            m_b_asr_block = b_in;
            continue;
        }

        auto original_port = ov::npuw::util::find_port_by_names(original_inputs, names);
        if (original_port.has_value()) {
            m_model_b_in_map.push_back({b_in, original_port.value()});
        } else {
            b_missing_ports.push_back(safe_any_name(b_in));
        }
    }
    throw_if_missing(b_missing_ports);

    // Initial setting of tensors for sub-requests
    for (const auto& item : m_model_a_in_map) {
        m_model_a_request->set_tensor(item.first, get_tensor(item.second));
    }
    // Feed static Model B inputs
    for (const auto& item : m_model_b_in_map) {
        m_model_b_request->set_tensor(item.first, get_tensor(item.second));
    }
}

void ov::npuw::KokoroInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                              const ov::SoPtr<ov::ITensor>& tensor) {
    ov::ISyncInferRequest::set_tensor(port, tensor);

    // If mappings are initialized, update sub-requests
    // Model A inputs
    for (const auto& item : m_model_a_in_map) {
        if (item.second == port) {
            m_model_a_request->set_tensor(item.first, tensor);
        }
    }
    // Model B inputs
    for (const auto& item : m_model_b_in_map) {
        if (item.second == port) {
            m_model_b_request->set_tensor(item.first, tensor);
        }
    }
}

void ov::npuw::KokoroInferRequest::infer() {
    OPENVINO_ASSERT(m_model_a_request);
    OPENVINO_ASSERT(m_model_b_request);

    const auto original_inputs = m_kokoro_compiled_model->inputs();
    const auto original_outputs = m_kokoro_compiled_model->outputs();

    // 1) Infer Model A
    m_model_a_request->infer();

    const auto pred_dur_tensor = m_model_a_request->get_tensor(m_a_pred_dur);
    const auto en_left_tensor = m_model_a_request->get_tensor(m_a_en_left);
    const auto asr_left_tensor = m_model_a_request->get_tensor(m_a_asr_left);
    OPENVINO_ASSERT(pred_dur_tensor && en_left_tensor && asr_left_tensor);

    auto orig_pred_dur = ov::npuw::util::find_port_by_name(original_outputs, "pred_dur");
    set_tensor(orig_pred_dur.value(), pred_dur_tensor);

    // 2) Build repeat-interleave indices

    std::vector<int64_t> pred;
    pred.resize(pred_dur_tensor->get_size());

    const auto et = pred_dur_tensor->get_element_type();
    if (et == ov::element::i64) {
        const auto* p = pred_dur_tensor->data<const int64_t>();
        std::copy_n(p, pred.size(), pred.data());
    } else if (et == ov::element::i32) {
        const auto* p = pred_dur_tensor->data<const int32_t>();
        std::copy_n(p, pred.size(), pred.data());
    } else {
        OPENVINO_THROW("Unexpected element type from pred_dur data, expected i64 or i32, got: ", et.get_type_name());
    }
    const std::size_t l_max = pred.size();

    std::size_t total_frames = 0;
    for (auto token_frames : pred) {
        if (token_frames > 0)
            total_frames += static_cast<std::size_t>(token_frames);
    }
    if (total_frames == 0) {
        OPENVINO_THROW("Sum(pred_dur) is zero; cannot generate audio");
    }

    std::vector<int64_t> idx_all;
    idx_all.reserve(total_frames);
    for (std::size_t i = 0; i < l_max; ++i) {
        const auto reps = pred[i];
        for (int64_t r = 0; r < reps; ++r) {
            idx_all.push_back(static_cast<int64_t>(i));
        }
    }
    OPENVINO_ASSERT(idx_all.size() == total_frames);

    // 3) Prepare Model B
    const auto b_model = m_model_b_request->get_compiled_model();
    const auto b_inputs = b_model->inputs();
    const auto b_outputs = b_model->outputs();

    OPENVINO_ASSERT(m_b_en_block.get_node(), "Kokoro Model B input 'en_block' not initialized");
    OPENVINO_ASSERT(m_b_asr_block.get_node(), "Kokoro Model B input 'asr_block' not initialized");

    // Pick audio output = first floating output (f16/f32)
    std::optional<ov::Output<const ov::Node>> audio_out;
    for (const auto& out : b_outputs) {
        const auto et = out.get_element_type();
        if (et == ov::element::f16 || et == ov::element::f32) {
            audio_out = out;
            break;
        }
    }

    if (!audio_out.has_value()) {
        OPENVINO_THROW("Kokoro Model B: no floating-point outputs found");
    }

    const size_t overlap_size = m_kokoro_compiled_model->overlap_size();
    const auto one_side_overlap = static_cast<std::size_t>(overlap_size / 2);

    // Block size will stay the same even if overlap is used, but effective input size per block decrease
    const std::size_t block = static_cast<std::size_t>(m_kokoro_compiled_model->block_size());
    OPENVINO_ASSERT(overlap_size < block, "NPUW_KOKORO_OVERLAP_SIZE must be smaller than block size");

    // Effective step size (how much we advance in the original sequence)
    // We reserve space for overlap on both sides: [overlap | step | overlap]
    // Total window size is 'block'.
    OPENVINO_ASSERT(one_side_overlap * 2 < block,
                    "NPUW_KOKORO_OVERLAP_SIZE is too large for the given block size (must be < block)");
    const std::size_t step = block - 2 * one_side_overlap;
    OPENVINO_ASSERT(step > 0, "Step size is zero, block size too small for overlap");

    const std::size_t num_blocks = (total_frames + step - 1) / step;

    std::vector<uint8_t> audio_bytes;
    std::size_t total_samples = 0;
    double samples_per_frame = 0.0;
    ov::Shape audio_shape0;
    ov::element::Type audio_type0;

    for (std::size_t blk = 0; blk < num_blocks; ++blk) {
        // Logical start of the new data in this block
        const std::size_t logical_start = blk * step;

        // Calculate input window [t0, t1)
        // We try to center the window: [logical_start - overlap, logical_start + step + overlap]
        // But constrained by [0, total_frames] and max length 'block'
        const std::size_t t0 = (blk == 0) ? 0 : (logical_start - one_side_overlap);
        std::size_t t1 = t0 + block;
        if (t1 > total_frames) {
            t1 = total_frames;
        }

        std::vector<int64_t> idx_block(idx_all.begin() + t0, idx_all.begin() + t1);

        auto en_in_tensor = m_model_b_request->get_tensor(m_b_en_block);
        auto asr_in_tensor = m_model_b_request->get_tensor(m_b_asr_block);
        OPENVINO_ASSERT(en_in_tensor && asr_in_tensor);

        // 3D tensor: [1, number_of_tokens, block_size]
        OPENVINO_ASSERT(en_in_tensor->get_shape().size() == 3u);
        OPENVINO_ASSERT(asr_in_tensor->get_shape().size() == 3u);
        OPENVINO_ASSERT(en_in_tensor->get_shape().back() == block);
        OPENVINO_ASSERT(asr_in_tensor->get_shape().back() == block);

        const auto en_et = en_in_tensor->get_element_type();
        const auto asr_et = asr_in_tensor->get_element_type();
        OPENVINO_ASSERT(en_et == en_left_tensor->get_element_type(), "en_block dtype mismatch with en_left");
        OPENVINO_ASSERT(asr_et == asr_left_tensor->get_element_type(), "asr_block dtype mismatch with asr_left");

        // Using idx_block fill inputs for model b (aligned feature matrix)
        if (en_et == ov::element::f16) {
            gather_3d_last_dim<ov::float16>(en_left_tensor, en_in_tensor->data<ov::float16>(), idx_block, block);
        } else if (en_et == ov::element::f32) {
            gather_3d_last_dim<float>(en_left_tensor, en_in_tensor->data<float>(), idx_block, block);
        } else {
            OPENVINO_THROW("en_left has unsupported type: ", en_et);
        }

        if (asr_et == ov::element::f16) {
            gather_3d_last_dim<ov::float16>(asr_left_tensor, asr_in_tensor->data<ov::float16>(), idx_block, block);
        } else if (asr_et == ov::element::f32) {
            gather_3d_last_dim<float>(asr_left_tensor, asr_in_tensor->data<float>(), idx_block, block);
        } else {
            OPENVINO_THROW("asr_left has unsupported type: ", asr_et);
        }

        m_model_b_request->infer();
        auto audio_tensor = m_model_b_request->get_tensor(audio_out.value());
        OPENVINO_ASSERT(audio_tensor);

        OPENVINO_ASSERT(!audio_tensor->get_shape().empty());
        const std::size_t block_samples = audio_tensor->get_shape().back();
        if (blk == 0) {
            audio_shape0 = audio_tensor->get_shape();
            audio_type0 = audio_tensor->get_element_type();
            if (samples_per_frame == 0.0) {
                // Estimate samples_per_frame. Model B is static, so it produces block_samples for 'block' input frames,
                // even if has less input data.
                samples_per_frame = static_cast<double>(block_samples) / static_cast<double>(block);
            }
        }

        // Calculate which part of the output audio is valid
        // We want to keep the audio corresponding to [logical_start, logical_start + step)
        // The input started at t0.
        // So we skip (logical_start - t0) frames.
        const std::size_t frames_to_skip = logical_start - t0;

        // We keep 'step' frames, but clamped to the end of the sequence
        std::size_t frames_to_keep = step;
        if (logical_start + frames_to_keep > total_frames) {
            frames_to_keep = total_frames - logical_start;
        }

        // Convert frames to samples
        const std::size_t skip_samples =
            static_cast<std::size_t>(std::llround(samples_per_frame * static_cast<double>(frames_to_skip)));
        std::size_t keep_samples =
            static_cast<std::size_t>(std::llround(samples_per_frame * static_cast<double>(frames_to_keep)));

        // Safety clamp
        if (skip_samples >= block_samples) {
            keep_samples = 0;
        } else if (skip_samples + keep_samples > block_samples) {
            keep_samples = block_samples - skip_samples;
        }

        const std::size_t elem_size = audio_tensor->get_element_type().size();
        const std::size_t bytes_to_copy = keep_samples * elem_size;
        const auto* src = static_cast<const uint8_t*>(audio_tensor->data());
        const auto* start_byte = src + skip_samples * elem_size;

        audio_bytes.insert(audio_bytes.end(), start_byte, start_byte + bytes_to_copy);
        total_samples += keep_samples;
    }

    if (original_outputs.empty()) {
        return;
    }
    auto out_shape = audio_shape0;
    if (!out_shape.empty()) {
        out_shape.back() = total_samples;
    } else {
        out_shape = ov::Shape{total_samples};
    }

    auto out_tensor = ov::make_tensor(audio_type0, out_shape);
    OPENVINO_ASSERT(out_tensor->get_byte_size() == audio_bytes.size());
    std::copy_n(audio_bytes.begin(), audio_bytes.size(), static_cast<uint8_t*>(out_tensor->data()));
    set_tensor(original_outputs[0], out_tensor);
}

ov::SoPtr<ov::ITensor> ov::npuw::KokoroInferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    return ov::ISyncInferRequest::get_tensor(port);
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::KokoroInferRequest::query_state() const {
    // FIXME Not implemented
    // OPENVINO_NOT_IMPLEMENTED;
    return {};
}
