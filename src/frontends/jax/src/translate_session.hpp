// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"
#include "openvino/frontend/extension/telemetry.hpp"
#include "openvino/frontend/jax/node_context.hpp"

namespace ov {
namespace frontend {
namespace jax {

class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::map<std::string, CreatorFunction>& translator_map,
                     const std::shared_ptr<TelemetryExtension>& telemetry);
    ~TranslateSession();
    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

    /// \brief Completely convert jax_model, creates JaxFrameworkNode if not possible to convert node
    /// \param jax_model Input model
    /// \param external_tensor_map Is used for recursive calls of convert_jax_model and represent the external
    /// context which is visible from nested model. Empty external_tensor_map is used as an indication that this is a
    /// main body conversion.
    /// \return fully converted OV Model
    std::shared_ptr<Model> convert_jax_model(std::shared_ptr<JaxDecoder> jax_model,
                                             const std::shared_ptr<jax::InputModel>& input_model = nullptr);

    /// \brief Writes jax tensor index into openvino tensor
    void encode_tensor_name(Output<Node> tensor_desc,
                            size_t tensor_idx,
                            std::vector<std::string> additional_names = {});

    /// \brief Gets jax tensor index from openvino tensor
    size_t decode_tensor_name(const Output<Node>& tensor_desc);

private:
    OutputVector convert_node(const NodeContext& context);

    const frontend::InputModel::Ptr m_input_model;
    const std::map<std::string, CreatorFunction>& m_translator_map;
    std::shared_ptr<TelemetryExtension> m_telemetry;
    std::shared_ptr<Model> m_ov_model;

    std::map<size_t, std::pair<size_t, Output<Node>>> m_counter_map;
    std::map<std::string, uint64_t> m_op_statistics;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
