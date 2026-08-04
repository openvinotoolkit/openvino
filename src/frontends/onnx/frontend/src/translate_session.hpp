// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <unordered_map>

#include "openvino/frontend/input_model.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace frontend {

class TelemetryExtension;

namespace onnx {

class OperatorsBridge;
class DecoderBaseOperation;
class TensorONNXPlace;

/// For one call of convert and decode method of Frontend, it creates one TranslateSession object to save data for the
/// translation session: telemetry statistics, cache of converted body graph models, operation translators (including
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

    /// \brief Converts the input model to an ov::Model.
    void translate_graph(const ov::frontend::InputModel::Ptr& input_model, std::shared_ptr<ov::Model>& ov_model);

    ov::frontend::InputModel::Ptr get_input_model(void) const {
        return m_input_model;
    }

    std::unordered_map<std::string, Output<ov::Node>>& get_tensor_values() {
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
    /// \brief Build a Result for a graph output (name carried onto the port, sink/friendly names set)
    /// and append it to results. Shared by both conversion paths.
    void add_result(const std::string& name, const ov::Output<ov::Node>& output_value, ResultVector& results);

    /// \brief Apply the registered operator translator (or the not-supported fallback) to one op
    /// decoder and return its OpenVINO outputs. Shared by both conversion paths.
    ov::OutputVector apply_op_translator(const std::shared_ptr<DecoderBaseOperation>& decoder,
                                         const std::shared_ptr<TelemetryExtension>& telemetry);

    /// \brief Translate an operation decoder and store its named outputs in m_tensor_values. Shared by
    /// both conversion paths (the op body is identical once inputs are materialized).
    void translate_op_and_store_outputs(const std::shared_ptr<DecoderBaseOperation>& op_decoder,
                                        const std::shared_ptr<TelemetryExtension>& telemetry);

    /// \brief Materialize a graph tensor as a Constant (data) or Parameter (no data), register it in
    /// m_tensor_values, append a created Parameter to m_parameters, and return the node.
    std::shared_ptr<ov::Node> create_const_or_param(const std::string& name,
                                                    const std::shared_ptr<TensorONNXPlace>& input_tensor);

    /// \brief Emit per-op "op_count" telemetry accumulated during the single-pass walk (the two-pass
    /// path emits it from load_model()). No-op when telemetry is null.
    void send_op_count_telemetry(const std::shared_ptr<TelemetryExtension>& telemetry,
                                 const std::map<std::string, uint64_t>& op_statistics) const;

    const ov::frontend::InputModel::Ptr m_input_model;
    const std::shared_ptr<OperatorsBridge> m_translator_map;
    const std::string m_model_name;
    std::shared_ptr<ov::Model> m_ov_model;
    std::unordered_map<std::string, Output<ov::Node>> m_tensor_values;
    bool m_fail_fast;
    TranslateSession* m_parent_session;
    ParameterVector m_parameters;
};
}  // namespace onnx
}  // namespace frontend
}  // namespace ov