// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/util/framework_node.hpp>

#include "utils.hpp"

#pragma once

namespace ov {
namespace frontend {
namespace pytorch {
class PtFrameworkNode : public ov::op::util::FrameworkNode {
public:
    OPENVINO_OP("PtFrameworkNode", "util", ::ov::op::util::FrameworkNode);

    PtFrameworkNode(const std::shared_ptr<Decoder>& decoder, const OutputVector& inputs, size_t output_size)
        : ov::op::util::FrameworkNode(inputs, output_size, decoder->get_subgraph_size()),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name("PTFrameworkNode");
        attrs["PtTypeName"] = m_decoder->get_op_type();
        attrs["PtSchema"] = m_decoder->get_schema();
        set_attrs(attrs);

        // Set output shapes and types if recognized
        for (size_t i = 0; i < output_size; ++i) {
            PartialShape ps;
            // TODO: Try to decode PT type as a custom type
            auto type = element::dynamic;
            if (i < decoder->num_of_outputs()) {
                try {
                    ps = m_decoder->get_output_shape(i);
                } catch (...) {
                    // nothing, means the info cannot be queried and remains unknown
                }
            }
            // TODO: Set custom `type` via special API
            set_output_type(i, type, ps);
        }
    }

    PtFrameworkNode(const std::shared_ptr<Decoder>& decoder, const OutputVector& inputs)
        : PtFrameworkNode(decoder, inputs, decoder->num_of_outputs()) {}

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto op = std::make_shared<PtFrameworkNode>(m_decoder, inputs, get_output_size());

        for (auto body_index = 0; body_index < m_bodies.size(); ++body_index) {
            op->set_function(body_index, clone_model(*get_function(body_index)));
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

    Decoder* get_decoder() const {
        return m_decoder.get();
    }

private:
    std::shared_ptr<Decoder> m_decoder;
};

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
