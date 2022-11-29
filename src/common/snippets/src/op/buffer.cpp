// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>

#include "snippets/op/buffer.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/runtime/host_tensor.hpp>

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ngraph::snippets::op::Buffer);

snippets::op::Buffer::Buffer(const Output<Node>& x) : Op({x}) {
    constructor_validate_and_infer_types();
}

bool snippets::op::Buffer::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Buffer_visit_attributes);
    visitor.on_attribute("offset", m_offset);
    return true;
}

std::shared_ptr<Node> snippets::op::Buffer::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Buffer_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    auto new_buffer = std::make_shared<Buffer>(new_args.at(0));
    new_buffer->set_offset(m_offset);
    return new_buffer;
}

void snippets::op::Buffer::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Buffer_validate_and_infer_types);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

void snippets::op::Buffer::set_offset(const size_t offset) {
    m_offset = offset;

    // If Buffer has offset We set this offset in the next Load and Store ops
    // to correctly read and write data because all buffers have the one register
    // Also if user sets offset to a Buffer It means that the Buffer has the corresponding Load and Store ops

    // Propagate to up: in Store. Buffer can have only one Store
    {
        auto parent = get_input_node_shared_ptr(0);
        auto idx = input(0).get_source_output().get_index();
        auto loop = std::dynamic_pointer_cast<snippets::op::LoopBase>(parent);
        while (loop) {
            parent = loop->get_input_node_shared_ptr(idx);
            idx = input(idx).get_source_output().get_index();
            loop = std::dynamic_pointer_cast<snippets::op::LoopBase>(parent);
        }
        if (auto store = std::dynamic_pointer_cast<snippets::op::Store>(parent)) {
            store->set_offset(m_offset);
        } else if (auto matmul = std::dynamic_pointer_cast<snippets::op::MatMulCPU>(parent)) {
            // MatMul encapsulates work with Loops inside himself
            matmul->set_offset_c(m_offset);
        } else {
            throw ngraph_error("Buffer::set_offset() was called when Buffer didn't have the corresponding Store op for offset propagation");
        }
    }

    // Propagate to down: in Load. Buffer can have several Load and Loops after himself. We should go through all target inputs
    {
        std::function<void(const Input<Node>&)> propagate_down;
        propagate_down = [&](const Input<Node>& target_input) {
            const auto child = target_input.get_node()->shared_from_this();
            if (std::dynamic_pointer_cast<snippets::op::LoopBase>(child)) {
                const auto index = target_input.get_index();
                for (const auto loop_target_output : child->output(index).get_target_inputs()) {
                    propagate_down(loop_target_output);
                }
            } else if (const auto load = std::dynamic_pointer_cast<snippets::op::Load>(child)) {
                load->set_offset(m_offset);
            } else if (auto matmul = std::dynamic_pointer_cast<snippets::op::MatMulCPU>(child)) {
                // MatMul encapsulates work with Loops inside himself
                if (target_input.get_index() == 0) {
                    matmul->set_offset_a(m_offset);
                } else {
                    matmul->set_offset_b(m_offset);
                }
            } else {
                throw ngraph_error("Buffer::set_offset() was called when Buffer didn't have the corresponding Load op for offset propagation");
            }
        };

        for (const auto target_output : output(0).get_target_inputs()) {
            propagate_down(target_output);
        }
    }
}
