// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

#pragma once

namespace ov {
namespace frontend {
namespace pytorch {
class PtFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("PtFrameworkNode", "util", ::ov::op::util::FrameworkNode);
    static constexpr const char* op_type_key = "PtTypeName";
    static constexpr const char* schema_key = "PtSchema";
    static constexpr const char* failed_conversion_key = "PtException";

    PtFrameworkNode(const std::shared_ptr<TorchDecoder>& decoder,
                    const OutputVector& inputs,
                    size_t output_size,
                    bool is_reverseprop = false,
                    bool skip_subgraphs = false)
        : ov::op::util::FrameworkNode(inputs, output_size, skip_subgraphs ? 0 : decoder->get_subgraph_size()),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name("PTFrameworkNode");
        if (is_reverseprop) {
            attrs[op_type_key] = m_decoder->get_op_type() + "_reverseprop";
            attrs[schema_key] = "None";
            attrs[failed_conversion_key] =
                "This is an internal openvino operation representing reverse data propagation. It should not appear in "
                "graph in normal conversion flow and might be result of other failures.";
        } else {
            attrs[op_type_key] = m_decoder->get_op_type();
            attrs[schema_key] = m_decoder->get_schema();
        }
        set_attrs(attrs);

        // Set output shapes and types if recognized
        for (size_t i = 0; i < output_size; ++i) {
            PartialShape ps;
            auto type = element::dynamic;
            if (i < decoder->num_of_outputs()) {
                try {
                    ps = m_decoder->get_output_shape(i);
                    auto dec_type = simplified_type_interpret(decoder->get_output_type(i));
                    if (dec_type.is<element::Type>())
                        type = dec_type.as<element::Type>();
                } catch (...) {
                    // nothing, means the info cannot be queried and remains unknown
                }
            }
            set_output_type(i, type, ps);
        }
    }

    PtFrameworkNode(const std::shared_ptr<TorchDecoder>& decoder, const OutputVector& inputs)
        : PtFrameworkNode(decoder, inputs, decoder->num_of_outputs()) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto op = std::make_shared<PtFrameworkNode>(m_decoder, inputs, get_output_size());

        for (size_t body_index = 0; body_index < m_bodies.size(); ++body_index) {
            op->set_function(static_cast<int>(body_index), get_function(static_cast<int>(body_index))->clone());
            for (const auto& m_input_descr : m_input_descriptions[body_index]) {
                op->m_input_descriptions[body_index].push_back(m_input_descr->copy());
            }
            for (const auto& m_output_descr : m_output_descriptions[body_index]) {
                op->m_output_descriptions[body_index].push_back(m_output_descr->copy());
            }
        }
        op->validate_and_infer_types();

        return op;
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    std::shared_ptr<TorchDecoder> get_decoder() const {
        return m_decoder;
    }

private:
    std::shared_ptr<TorchDecoder> m_decoder;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
