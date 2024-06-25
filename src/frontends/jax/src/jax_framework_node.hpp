// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/framework_node.hpp"
#include "utils.hpp"

#pragma once

namespace ov {
namespace frontend {
namespace jax {
class JaxFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("JaxFrameworkNode", "util", ::ov::op::util::FrameworkNode);
    static constexpr const char* op_type_key = "JaxTypeName";
    static constexpr const char* failed_conversion_key = "JaxException";

    JaxFrameworkNode(const std::shared_ptr<JaxDecoder>& decoder, const OutputVector& inputs, size_t output_size)
        : ov::op::util::FrameworkNode(inputs, output_size, 0),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name("JaxFrameworkNode");
        attrs[op_type_key] = m_decoder->get_op_type();
        set_attrs(attrs);

        // Set output shapes and types if recognized
        for (size_t i = 0; i < output_size; ++i) {
            PartialShape ps;
            auto type = element::dynamic;
            if (i < decoder->num_outputs()) {
                try {
                    ps = m_decoder->get_output_shape(i);
                    auto dec_type = decoder->get_output_type(i);
                    if (dec_type.is<element::Type>())
                        type = dec_type.as<element::Type>();
                } catch (...) {
                    // nothing, means the info cannot be queried and remains unknown
                }
            }
            set_output_type(i, type, ps);
        }
    }

    JaxFrameworkNode(const std::shared_ptr<JaxDecoder>& decoder, const OutputVector& inputs)
        : JaxFrameworkNode(decoder, inputs, decoder->num_outputs()) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto op = std::make_shared<JaxFrameworkNode>(m_decoder, inputs, get_output_size());

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

    std::shared_ptr<JaxDecoder> get_decoder() const {
        return m_decoder;
    }

private:
    std::shared_ptr<JaxDecoder> m_decoder;
};

}  // namespace jax
}  // namespace frontend
}  // namespace ov
