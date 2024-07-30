// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class FullyConnectedNode : public ov::op::Op {
public:
    OPENVINO_OP("FullyConnected", "cpu_plugin_opset");

    FullyConnectedNode() = default;

    template <typename... Args>
    FullyConnectedNode(const ov::Output<Node>& A,
                       const ov::Output<Node>& B,
                       const ov::Rank& output_rank,
                       const ov::element::Type output_type = ov::element::undefined,
                       const Args&... args)
        : Op({A, B}),
          m_output_rank(output_rank),
          m_output_type(output_type) {
        process_arguments(get_input_size(), args...);
        validate_and_infer_types();
    }

    // Base case: no arguments left to process
    void process_arguments(size_t) {
        // Do nothing
    }

    // Recursive case: process one argument and recurse for the rest
    template<typename T, typename... Args>
    void process_arguments(size_t index, const T& first, const Args&... rest) {
        set_argument(index, first);
        process_arguments(index + 1, rest...); // Recursive call
    }

    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    ov::Rank get_output_rank() const { return m_output_rank; }
    ov::element::Type get_output_type() const { return m_output_type; }
    bool has_subtract() const {
        return get_input_size() > 3;
    }

private:
    ov::Rank m_output_rank;
    ov::element::Type m_output_type;
};

}   // namespace intel_cpu
}   // namespace ov
