// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"

namespace ov {
namespace frontend {
namespace onnx {

class OperatorsBridge;

/// For one call of convert and decode method of Frontend, it creates one TranslateSession object to save data for the
/// translation session: telemetry statistics, cache of convrted body graph models, operation translators (including
/// extensions) registered for this translation session.
class TranslateSession {
public:
    TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                     const std::shared_ptr<OperatorsBridge>& translator_map,
                     const std::string& model_name);
    std::shared_ptr<ov::Model> get_converted_model();

    void translate_graph(const ov::frontend::InputModel::Ptr& input_model, std::shared_ptr<ov::Model>& ov_model);

    ov::frontend::InputModel::Ptr get_input_model(void) const {
        return m_input_model;
    }

    std::map<std::string, Output<ov::Node>>& get_tensor_values() {
        return m_tensor_values;
    }

private:
    const ov::frontend::InputModel::Ptr m_input_model;
    const std::shared_ptr<OperatorsBridge> m_translator_map;
    const std::string m_model_name;
    std::shared_ptr<ov::Model> m_ov_model;
    std::map<std::string, Output<ov::Node>> m_tensor_values;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov