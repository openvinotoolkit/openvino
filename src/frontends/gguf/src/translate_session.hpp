// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"
#include "node_context.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"

namespace ov::frontend::gguf {

void prepare_graph_inputs(TensorMap& values, GgufDecoder& decoder);
void translate_node(const std::shared_ptr<GgufDecoder>& decoder,
                    std::shared_ptr<TensorMap>& values,
                    const std::unordered_map<std::string, CreatorFunction>& translators,
                    const std::shared_ptr<ConstantCache>& constant_cache = {});

class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::unordered_map<std::string, CreatorFunction>& translator_map,
                     const std::vector<DecoderTransformationExtension::Ptr>& transformation_extensions = {});

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);
    // MTP draft converted with the model, sharing its weight buffers; null without one.
    const std::shared_ptr<Model>& get_mtp_draft_model() const {
        return m_mtp_draft;
    }

private:
    std::shared_ptr<Model> assemble_model(const std::shared_ptr<GgufDecoder>& decoder);
    std::shared_ptr<Model> finalize_model(std::shared_ptr<Model> model, const GgufDecoder& decoder);
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    const frontend::InputModel::Ptr m_input_model;
    const std::unordered_map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    std::shared_ptr<Model> m_mtp_draft;
    std::vector<DecoderTransformationExtension::Ptr> m_transformation_extensions;
};

}  // namespace ov::frontend::gguf
