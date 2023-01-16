// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {
using CachedBodyModelsType = std::unordered_map<std::string, std::shared_ptr<const ov::Model>>;
using TelemetryDataType = std::vector<std::pair<std::string, std::string>>;

/// For one call of convert and decode method of Frontend, it creates one TranslateSession object to save data for the
/// translation session: telemetry statistics, cache of convrted body graph models, operation translators (including
/// extensions) registered for this translation session.
class TranslateSession {
public:
    TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                     const std::shared_ptr<TranslatorDictionaryType>& translator_map,
                     const std::string& model_name,
                     bool fail_fast,
                     bool telemetry);
    std::shared_ptr<ov::Model> get_converted_model();
    std::shared_ptr<TelemetryDataType> get_telemetry_data() const;

    void translate_graph(const ov::frontend::InputModel::Ptr& input_model, std::shared_ptr<ov::Model>& ov_model);

    void inject_body_model(std::shared_ptr<ov::Model> body_model,
                           const std::string& operation_type,
                           const ov::OutputVector& ov_inputs,
                           ov::OutputVector& ov_outputs);

    std::shared_ptr<ov::Model> get_body_ov_model(const std::string& body_graph_name);

private:
    const ov::frontend::InputModel::Ptr m_input_model;
    const bool m_fail_fast;
    const bool m_telemetry;
    const std::shared_ptr<TranslatorDictionaryType> m_translator_map;
    const std::string m_model_name;

    std::shared_ptr<CachedBodyModelsType> m_cached_body_models;
    std::shared_ptr<TelemetryDataType> m_telemetry_data;
    std::shared_ptr<ov::Model> m_ov_model;

    void update_cached_body_models(const std::string& operation_type,
                                   const std::shared_ptr<const ov::Model>& cached_body_model) {
        m_cached_body_models->insert(std::make_pair(operation_type, cached_body_model));
    }
};

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
