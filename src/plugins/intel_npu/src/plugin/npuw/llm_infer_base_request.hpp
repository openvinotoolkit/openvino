// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <regex>

#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

class LLMInferBaseRequest : public ov::ISyncInferRequest {
public:
    struct layer_names {
        static constexpr const char* input_ids = "input_ids";
        static constexpr const char* inputs_embeds = "inputs_embeds";
        static constexpr const char* attention_mask = "attention_mask";
        static constexpr const char* position_ids = "position_ids";
        static constexpr const char* past_key_values = "past_key_values";
        static constexpr const char* output_embeds = "npuw_output_embed";
        static constexpr const char* logits = "logits";
        static constexpr const char* token_type_ids = "token_type_ids";
        static constexpr const char* gemma_sliding_mask = "npuw_gemma_sliding_mask";
    };

    struct layer_ids {
        static constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;
        static constexpr std::size_t kStartOutputKVCacheLayers = 1;
    };

    explicit LLMInferBaseRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model)
        : ISyncInferRequest(compiled_model),
          m_npuw_llm_compiled_model(compiled_model) {}

    void check_tensors() const override {};
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

protected:
    void update_kvcache_for(std::shared_ptr<ov::IAsyncInferRequest> request,
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
            const auto& input_name =
                std::regex_replace(output_name, std::regex("present"), layer_names::past_key_values);
            if (in_ports.find(input_name) == in_ports.end()) {
                // FIXME: Totally wrong debug message. input_name is an invalid name of input layer.
                LOG_DEBUG("Input name " << input_name << " doesn't contain kv cache. Skipping.");
                continue;
            }
            auto dst_tensor = request->get_tensor(in_ports.at(input_name));
            const auto& kv_dim =
                (output_name.find("value") != std::string::npos && v_transposed) ? 3u : kvcache_desc.dim;
            auto dst_slice = uu::make_tensor_slice(dst_tensor,
                                                   kv_dim,
                                                   kvcache_desc.num_stored_tokens - num_tokens,
                                                   kvcache_desc.num_stored_tokens);
            auto src_tensor = request->get_tensor(out_ports.at(output_name));

            // NOTE: Sometimes present kv layer can contain greater seq_len
            //       than was sent to be processed
            uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
            OPENVINO_ASSERT(num_tokens <= src_seq_len);
            if (src_seq_len > num_tokens) {
                auto src_slice = uu::make_tensor_slice(src_tensor, kv_dim, src_seq_len - num_tokens, src_seq_len);
                uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
            } else {
                uu::copy_tensor_by_dim(src_tensor, dst_slice, kv_dim, kv_dim);
            }
        }
    }

    void init_tensor(const ov::Output<const ov::Node>& port) {
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

    void init_ports() {
        for (const auto& input_port : m_npuw_llm_compiled_model->inputs()) {
            init_tensor(input_port);
        }
        for (const auto& output_port : m_npuw_llm_compiled_model->outputs()) {
            init_tensor(output_port);
        }
    }

protected:
    std::shared_ptr<LLMCompiledModel> m_npuw_llm_compiled_model;
};

}  // namespace npuw
}  // namespace ov
