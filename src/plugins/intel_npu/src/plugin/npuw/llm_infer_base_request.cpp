// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_infer_base_request.hpp"

#include <regex>

#include "infer_request_utils.hpp"

void ov::npuw::LLMInferBaseRequest::update_kvcache_for(
    std::shared_ptr<ov::IAsyncInferRequest> request,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
    uint32_t num_tokens,
    bool v_transposed) {
    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_npuw_llm_compiled_model->m_kvcache_desc;
    auto& compiled = request->get_compiled_model();
    // FIXME: Find only matching by names outputs and copy them, having previously checked that such inputs exist
    for (std::size_t i = layer_ids::kStartOutputKVCacheLayers; i < compiled->outputs().size(); ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();
        const auto& input_name = std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
        if (in_ports.find(input_name) == in_ports.end()) {
            // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
            LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
            continue;
        }
        auto dst_tensor = request->get_tensor(in_ports.at(input_name));
        const auto& kv_dim = (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kvcache_desc.dim;
        auto src_tensor = request->get_tensor(out_ports.at(output_name));

        // NOTE: Sometimes present kv layer can contain greater seq_len
        //       than was sent to be processed
        const uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
        OPENVINO_ASSERT(num_tokens <= src_seq_len);
        const uint32_t src_start = src_seq_len - num_tokens;
        const uint32_t dst_start = kvcache_desc.num_stored_tokens - num_tokens;
        const uint32_t dst_end   = kvcache_desc.num_stored_tokens;

        // i4/u4 tensors cannot be sliced via ROI tensor (OV runtime does not support
        // sub-byte strides). Use the nibble-aware direct copy for these types.
        const bool is_i4 = (dst_tensor->get_element_type().bitwidth() == 4u);
        if (is_i4) {
            uu::copy_tensor_slice_i4(src_tensor, kv_dim, src_start, src_seq_len,
                                     dst_tensor, kv_dim, dst_start);
        } else {
            auto dst_slice = uu::make_tensor_slice(dst_tensor, kv_dim, dst_start, dst_end);
            if (src_start > 0u) {
                auto src_slice = uu::make_tensor_slice(src_tensor, kv_dim, src_start, src_seq_len);
                uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
            } else {
                uu::copy_tensor_by_dim(src_tensor, dst_slice, kv_dim, kv_dim);
            }
        }
    }
}

void ov::npuw::LLMInferBaseRequest::init_tensor(const ov::Output<const ov::Node>& port) {
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

void ov::npuw::LLMInferBaseRequest::init_ports() {
    for (const auto& input_port : m_npuw_llm_compiled_model->inputs()) {
        init_tensor(input_port);
    }
    for (const auto& output_port : m_npuw_llm_compiled_model->outputs()) {
        init_tensor(output_port);
    }
}
