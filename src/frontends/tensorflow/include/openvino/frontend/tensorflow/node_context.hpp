// Copyright (C) 2018-2023 Intel Corporation
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
class TranslateSession;

/// Keep necessary data for a single node in the original FW graph to facilitate
/// conversion process in the rules code.
class NodeContext : public ov::frontend::NodeContext {
public:
    using Ptr = std::shared_ptr<NodeContext>;
    NodeContext(const std::shared_ptr<DecoderBase>& decoder,
                const OutputVector& inputs,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type()),
          m_decoder(decoder),
          m_translate_session(translate_session),
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
    const std::string& get_name() const override {
        return m_decoder->get_op_name();
    }

    /// \brief Get a decoder
    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        auto res = m_decoder->get_attribute(name);
        return res;
    }

    /// \brief Get a pointer to TranslateSession object
    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

private:
    ov::Any apply_additional_conversion_rules(const ov::Any& data, const std::type_info& type_info) const override;

    std::shared_ptr<DecoderBase> m_decoder;
    TranslateSession* m_translate_session;
    const OutputVector& m_inputs;
};

using CreatorFunctionIndexed = std::function<ov::OutputVector(const ov::frontend::tensorflow::NodeContext&)>;
using CreatorFunctionNamedAndIndexed = std::function<NamedOutputVector(const ov::frontend::tensorflow::NodeContext&)>;

class CreatorFunction {
public:
    CreatorFunction() = default;
    CreatorFunction(CreatorFunctionIndexed _func) : func_indexed(_func) {}
    CreatorFunction(CreatorFunctionNamedAndIndexed _func) : func_named_and_indexed(_func) {}

    NamedOutputVector operator()(const ov::frontend::tensorflow::NodeContext& node) const {
        if (func_indexed) {
            auto outputs = func_indexed(node);
            return NamedOutputVector(outputs.begin(), outputs.end());
        } else if (func_named_and_indexed) {
            return func_named_and_indexed(node);
        } else {
            FRONT_END_GENERAL_CHECK(false, "No conversion function exist in this CreatorFunction");
        }
    }

private:
    CreatorFunctionIndexed func_indexed = nullptr;
    CreatorFunctionNamedAndIndexed func_named_and_indexed = nullptr;
};

using TranslatorDictionaryType = std::map<std::string, CreatorFunction>;

}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
