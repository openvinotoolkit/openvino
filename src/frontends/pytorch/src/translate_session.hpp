// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "input_model.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

/// For one call of convert and decode method of Frontend, it creates one TranslateSession object to save data for the
/// translation session: telemetry statistics, operation translators (including extensions) registered for this
/// translation session.
class TranslateSession {
public:
    TranslateSession(const frontend::InputModel::Ptr& input_model,
                     const std::map<std::string, PytorchCreatorFunction>& translator_map);
    std::shared_ptr<Model> get_converted_model();
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr& input_model);

    /// \brief Completely convert pytorch_model, creates PtFrameworkNode if not possible to convert node
    /// \param pytorch_model Input model
    /// \param external_tensor_map Is used for recursive calls of convert_pytorch_model and represent the external
    /// context which is visible from nested model. Empty external_tensor_map is used as an indication that this is a
    /// main body conversion.
    /// \return fully converted OV Model
    std::shared_ptr<Model> convert_pytorch_model(
        std::shared_ptr<TorchDecoder> pytorch_model,
        const TensorMap& external_tensor_map = {},
        const std::unordered_map<size_t, PlaceDesc>& external_descriptors = {});

    void encode_tensor_name(Output<Node> tensor_desc, size_t tensor_idx, std::string debug_name = "");
    size_t decode_tensor_name(const Output<Node>& tensor_desc);

    size_t m_friendly_name_counter = 0;

private:
    OutputVector convert_node(NodeContext& context);

    const frontend::InputModel::Ptr m_input_model;
    const std::map<std::string, PytorchCreatorFunction>& m_translator_map;

    std::shared_ptr<Model> m_ov_model;
    std::map<size_t, std::pair<size_t, Output<Node>>> m_counter_map;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
