// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/frontend/tensorflow/variables_map.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

struct CachedBodyModelSignature {
    std::string body_name;
    std::vector<ov::PartialShape> input_shapes;
    std::vector<ov::element::Type> input_types;

    bool operator==(const CachedBodyModelSignature& other) const {
        return (body_name == other.body_name && input_shapes == other.input_shapes && input_types == other.input_types);
    }
};

struct CachedBodyModelSignatureHasher {
    std::size_t operator()(const CachedBodyModelSignature& k) const {
        return std::hash<std::string>()(k.body_name);
    }
};

using CachedBodyModelsType =
    std::unordered_map<CachedBodyModelSignature, std::shared_ptr<const ov::Model>, CachedBodyModelSignatureHasher>;

/// For one call of convert and decode method of Frontend, it creates one TranslateSession object to save data for the
/// translation session: telemetry statistics, cache of convrted body graph models, operation translators (including
/// extensions) registered for this translation session.
class TranslateSession {
public:
    TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                     const std::shared_ptr<TranslatorDictionaryType>& translator_map,
                     const std::string& model_name);
    std::shared_ptr<ov::Model> get_converted_model();

    void translate_graph(const ov::frontend::InputModel::Ptr& input_model, std::shared_ptr<ov::Model>& ov_model);

    std::shared_ptr<ov::Model> get_body_ov_model(const std::string& body_graph_name,
                                                 const ov::OutputVector& ov_inputs,
                                                 bool clear_names = true);

    ov::frontend::InputModel::Ptr get_input_model(void) const {
        return m_input_model;
    }

private:
    const ov::frontend::InputModel::Ptr m_input_model;
    const std::shared_ptr<TranslatorDictionaryType> m_translator_map;
    const std::string m_model_name;
    std::shared_ptr<ov::Model> m_ov_model;

    // this is a container to cache already converted body graph models for operations
    // such as While, If, PartitionedCall.
    // the caching happens by body graph name (or function name) and input shapes and types
    // specified for its conversion.
    // the same topology can be converted with different shapes and types so it will be cached separately
    std::shared_ptr<CachedBodyModelsType> m_cached_body_models;

    // stores variables states at each node of the graph
    VariableMap::Ptr m_variables_map;

    void update_cached_body_models(const CachedBodyModelSignature& cached_body_model_signature,
                                   const std::shared_ptr<const ov::Model>& cached_body_model) {
        m_cached_body_models->insert(std::make_pair(cached_body_model_signature, cached_body_model));
    }

    VariableMap::Ptr get_variable_map(void) const {
        return m_variables_map;
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
