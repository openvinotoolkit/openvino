// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "decoder.hpp"
#include "exception.hpp"
#include "openvino/core/any.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    NodeContext(const DecoderBase& decoder, const OutputVector& inputs)
        : ov::frontend::NodeContext(decoder.get_op_type()),
          m_decoder(decoder),
          m_inputs(inputs) {}

    /// Detects if there is at least one input attached with a given name
    bool has_input(const size_t& port_index) const {
        return port_index < m_inputs.size();
    }

    Output<Node> get_input(int port_index) const override {
        return m_inputs.at(port_index);
    }

    size_t get_input_size() const override {
        return m_inputs.size();
    }

    /// \brief Get a node name
    std::string get_name() const {
        return m_decoder.get_op_name();
    }

    /// \brief Get a decoder
    const DecoderBase* get_decoder() const {
        return &m_decoder;
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        auto res = m_decoder.get_attribute(name);
        return res;
    }

private:
    ov::Any apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const override;

    const DecoderBase& m_decoder;
    const OutputVector& m_inputs;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::tensorflow::NodeContext&)>;
using TranslatorDictionaryType = std::map<std::string, CreatorFunction>;

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
