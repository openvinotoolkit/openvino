// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/propagate_buffer_offset.hpp"
#include "snippets/op/subgraph.hpp"


ngraph::snippets::pass::PropagateBufferOffset::PropagateBufferOffset() {
    MATCHER_SCOPE(PropagateBufferOffset);

    auto m_buffer = ngraph::pattern::wrap_type<op::Buffer>();

    auto callback = [&](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::PropagateBufferOffset")
        auto root = m.get_match_root();
        const auto buffer = ov::as_type_ptr<op::Buffer>(root);

        // If Buffer has offset We set this offset in the next Load and Store ops
        // to correctly read and write data because all buffers have the one register
        // Also if user sets offset to a Buffer It means that the Buffer has the corresponding Load and Store ops

        // Propagate to up: in Store. Buffer can have only one Store
        {
            auto parent = buffer->get_input_node_shared_ptr(0);
            auto idx = buffer->input(0).get_source_output().get_index();
            while (std::dynamic_pointer_cast<snippets::op::LoopBase>(parent)) {
                const auto source_output = parent->input_value(idx);
                parent = source_output.get_node_shared_ptr();
                idx = source_output.get_index();
            }
            if (auto store = std::dynamic_pointer_cast<snippets::op::Store>(parent)) {
                store->set_offset(current_offset);
            } else if (const auto brgemm = std::dynamic_pointer_cast<snippets::op::Brgemm>(parent)) {
                // Brgemm encapsulates work with loading and storing of data
                brgemm->set_offset_c(current_offset);
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
                    load->set_offset(current_offset);
                } else if (const auto brgemm = std::dynamic_pointer_cast<snippets::op::Brgemm>(child)) {
                    // Brgemm encapsulates work with loading and storing of data
                    if (target_input.get_index() == 0) {
                        brgemm->set_offset_a(current_offset);
                    } else {
                        brgemm->set_offset_b(current_offset);
                    }
                } else {
                    throw ngraph_error("Buffer::set_offset() was called when Buffer didn't have the corresponding Load op for offset propagation");
                }
            };

            for (const auto target_output : buffer->output(0).get_target_inputs()) {
                propagate_down(target_output);
            }
        }

        current_offset += buffer->get_byte_size();
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_buffer, matcher_name);
    register_matcher(m, callback);
}
