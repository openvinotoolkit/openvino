// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/util/framework_node.hpp>

#include "decoder.hpp"

namespace ngraph {
namespace frontend {
class PDPDFrameworkNode : public ov::op::util::FrameworkNode {
public:
    NGRAPH_RTTI_DECLARATION;

    PDPDFrameworkNode(const DecoderPDPDProto& decoder,
                      const OutputVector& inputs,
                      const std::vector<std::string>& inputs_names)
        : FrameworkNode(inputs, decoder.get_output_size()),
          m_decoder{decoder},
          m_inputs_names{inputs_names} {
        ov::op::util::FrameworkNodeAttrs attrs;
        attrs.set_type_name(m_decoder.get_op_type());
        set_attrs(attrs);

        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& inputs) const override {
        return std::make_shared<PDPDFrameworkNode>(m_decoder, inputs, m_inputs_names);
    }

    std::string get_op_type() const {
        return m_decoder.get_op_type();
    }

    const DecoderPDPDProto& get_decoder() const {
        return m_decoder;
    }

    std::map<std::string, OutputVector> get_named_inputs() const;

    std::map<std::string, OutputVector> return_named_outputs();

private:
    const DecoderPDPDProto m_decoder;
    std::vector<std::string> m_inputs_names;
};
}  // namespace frontend
}  // namespace ngraph
