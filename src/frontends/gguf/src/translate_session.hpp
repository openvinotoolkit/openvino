// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"
#include "node_context.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"

namespace ov {
namespace frontend {
namespace gguf {

class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::unordered_map<std::string, CreatorFunction>& translator_map,
                     const std::vector<DecoderTransformationExtension::Ptr>& transformation_extensions = {});

    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

private:
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);
    const frontend::InputModel::Ptr m_input_model;
    const std::unordered_map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    std::vector<DecoderTransformationExtension::Ptr> m_transformation_extensions;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
