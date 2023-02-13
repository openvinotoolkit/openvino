// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_buffer.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::InsertBuffer::InsertBuffer(const int32_t allocation_rank) {
    MATCHER_SCOPE(InsertBuffer);
    // The list of operations that require Buffers on their Inputs and Outputs
    const auto pattern = ngraph::pattern::wrap_type<ngraph::op::v1::Softmax,
                                                    ngraph::op::v8::Softmax,
                                                    ngraph::op::v1::Transpose,
                                                    op::Brgemm>();

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(pattern, matcher_name),
            [this, allocation_rank](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertBuffer")
            auto root = m.get_match_root();
            bool rewritten = false;

            // check if already has Buffer, Parameter or Constant as an input
            for (const auto& input : root->inputs()) {
                const auto input_node = input.get_source_output().get_node()->shared_from_this();
                if (!ov::is_type<ngraph::snippets::op::Buffer>(input_node) &&
                    !ov::is_type<ngraph::op::v0::Parameter>(input_node) &&
                    !ov::is_type<ngraph::op::v0::Constant>(input_node)) {
                    const auto buffer = std::make_shared<ngraph::snippets::op::Buffer>(input_node, allocation_rank);
                    root->set_argument(input.get_index(), buffer);
                    rewritten |= true;
                }
                if (ov::is_type<op::Buffer>(input.get_source_output().get_node_shared_ptr()) &&
                    input.get_source_output().get_target_inputs().size() != 1) {
                    throw ngraph::ngraph_error(
                            "If Buffer is a input for operation output, this Buffer should be a single consumer for this port");
                }
            }

            // check if already has Buffer or outputs is Result
            for (const auto& output : root->outputs()) {
                const auto target_inputs = output.get_target_inputs();
                if (target_inputs.size() > 1) {
                    for (const auto& consumer : target_inputs) {
                        const auto output_node = consumer.get_node()->shared_from_this();
                        if (ov::is_type<ngraph::snippets::op::Buffer>(output_node)) {
                            // If some of children from one common port are different Buffers,
                            // we should remove them to insert one common Buffer on one common port
                            replace_output_update_name(output_node->output(0), output_node->input_value(0));
                        } else if (ov::is_type<ngraph::op::v0::Result>(output_node)) {
                            // TODO: At this moment operation which is should be wrapped by Buffers doesn't support several childs where one of them is Result
                            // because Result and Buffer from one root port should have the same register. It's not supported at the moment
                            // For example,
                            //    Buffer
                            //      |
                            //    Softmax
                            //    /    \
                            // Buffer Result
                            throw ngraph::ngraph_error(
                                "Operation which is should be wrapped by Buffers has few children from one output port where one of them is Result");
                        }
                    }
                }

                const auto buffer = std::make_shared<ngraph::snippets::op::Buffer>(output, allocation_rank);
                for (const auto& consumer : output.get_target_inputs()) {
                    const auto output_node = consumer.get_node()->shared_from_this();
                    if (output_node != buffer &&
                        !ov::is_type<ngraph::snippets::op::Buffer>(output_node) &&
                        !ov::is_type<ngraph::op::v0::Result>(output_node)) {
                        consumer.replace_source_output(buffer);
                        rewritten |= true;
                    }
                }

                const auto new_target_inputs = output.get_target_inputs();
                const auto has_buffer_on_output = std::any_of(new_target_inputs.begin(), new_target_inputs.end(), [](const ov::Input<ov::Node>& consumer) {
                    const auto child = consumer.get_node()->shared_from_this();
                    // We check for count of target inputs of Buffer output because
                    // we created Buffer op with root input previously for the next possible insertions
                    // Thus, if Buffer wasn't inserted, this op doesn't have target inputs on output
                    return ov::is_type<ngraph::snippets::op::Buffer>(child) && child->output(0).get_target_inputs().size() > 0;
                });
                if (has_buffer_on_output && new_target_inputs.size() != 1) {
                    throw ngraph::ngraph_error(
                            "If Buffer is a input for operation output, this Buffer should be a single consumer for this port");
                }
            }
            return rewritten;
        });
}
