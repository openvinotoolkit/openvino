// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/input_model.hpp"
#include "openvino/op/parameter.hpp"

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
    TranslateSession(const ov::frontend::InputModel::Ptr& input_model,
                     TranslateSession* parent_session,
                     const std::string& model_name);

    std::shared_ptr<ov::Model> get_converted_model();

    void translate_graph(const ov::frontend::InputModel::Ptr& input_model, std::shared_ptr<ov::Model>& ov_model);

    ov::frontend::InputModel::Ptr get_input_model(void) const {
        return m_input_model;
    }

    std::map<std::string, Output<ov::Node>>& get_tensor_values() {
        return m_tensor_values;
    }

    void set_fail_fast(const bool fail_fast) {
        m_fail_fast = fail_fast;
    }

    bool get_fail_fast() const {
        return m_fail_fast;
    }

    /// \brief Method tries to find an output which is named "name"
    /// If a result found in a parent session (it means current GraphIterator/InputModel
    /// doesn't have a corresponding node. And if node isn't a constant - then
    /// method restores a Parameters chain by adding a Parameter with provided name
    ov::Output<ov::Node> lookup_tensor(const std::string& name);

private:
    const ov::frontend::InputModel::Ptr m_input_model;
    const std::shared_ptr<OperatorsBridge> m_translator_map;
    const std::string m_model_name;
    std::shared_ptr<ov::Model> m_ov_model;
    std::map<std::string, Output<ov::Node>> m_tensor_values;
    bool m_fail_fast;
    TranslateSession* m_parent_session;
    ParameterVector m_parameters;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov