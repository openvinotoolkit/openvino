// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include "openvino/frontend/decoder.hpp"
#include "openvino/op/op.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

class InternalOpDecoder : public DummyDecoder {
public:
    explicit InternalOpDecoder(const std::string& op_type, const size_t num_outputs)
        : m_op_type(op_type),
          m_num_outputs(num_outputs) {}
    const std::string& get_op_type() const override {
        return m_op_type;
    }
    size_t num_of_outputs() const override {
        return m_num_outputs;
    }
    size_t get_subgraph_size() const override {
        return 0;
    }
    const std::string& decoder_type_name() const override {
        return m_decoder_type;
    }

private:
    const std::string m_op_type;
    const std::string m_decoder_type = "internal_op";
    const size_t m_num_outputs;
};

class InternalOperation : public PtFrameworkNode {
public:
    OPENVINO_OP("InternalOperation", "util", PtFrameworkNode);

protected:
    InternalOperation(const std::string& op_type,
                      const OutputVector& inputs,
                      size_t num_outputs,
                      const std::string& no_conversion_reason)
        : PtFrameworkNode(std::make_shared<InternalOpDecoder>(op_type, num_outputs), inputs) {
        auto attrs = get_attrs();
        attrs[PtFrameworkNode::failed_conversion_key] = no_conversion_reason;
        set_attrs(attrs);
    }
};

class InternalReverseOperation : public ov::op::Op {
public:
    OPENVINO_OP("InternalReverseOperation", "internal");
    InternalReverseOperation(const OutputVector& inputs) : ov::op::Op(inputs) {}
};
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
