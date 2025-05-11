// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/frontend/decoder.hpp"
#include "openvino/op/util/framework_node.hpp"

namespace ov {
namespace frontend {
namespace tensorflow {

class FrameworkNode : public ov::op::util::FrameworkNode {
public:
    static constexpr const char* failed_conversion_key = "tensorflow::FrameworkNode::failed_conversion_key";
    OPENVINO_OP("TFFrameworkNode", "util", ::ov::op::util::FrameworkNode);

    FrameworkNode(const std::shared_ptr<DecoderBase>& decoder, const OutputVector& inputs, size_t num_outputs)
        : ov::op::util::FrameworkNode(inputs, std::max(num_outputs, size_t(1))),
          m_decoder(decoder) {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name(m_decoder->get_op_type());
        set_attrs(attrs);

        validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        for (size_t i = 0; i < get_output_size(); ++i) {
            set_output_type(i, ov::element::dynamic, PartialShape::dynamic());
        }
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        auto fw_node = std::make_shared<FrameworkNode>(m_decoder, inputs, get_output_size());
        fw_node->set_attrs(get_attrs());
        return fw_node;
    }

    std::string get_op_type() const {
        return m_decoder->get_op_type();
    }

    std::shared_ptr<DecoderBase> get_decoder() const {
        return m_decoder;
    }

protected:
    std::shared_ptr<DecoderBase> m_decoder;
};
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
