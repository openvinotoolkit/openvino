// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_load_store.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::InsertLoad::InsertLoad(const size_t count) {
    MATCHER_SCOPE(InsertLoad);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Parameter, ngraph::snippets::op::Buffer>(), matcher_name),
            [this, count](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertLoad")
            auto root = m.get_match_root();

            // check if already has Load as an output
            for (const auto& output : root->outputs()) {
                for (const auto& consumer : output.get_target_inputs()) {
                    // if a parameter is connected to a Load => we don't need another one
                    // if a parameter is connected to LoopBegin => there must be Load inside the Loop
                    // if a parameter is connected to MatMul => we don't need Load (read/write is encapsulated into the brgemm emitter)
                    // (it's the responsibility of transformation that inserted the Loops)
                    const auto& consumer_node = consumer.get_node();
                    if (ov::is_type<ngraph::snippets::op::Load>(consumer_node) ||
                        ov::is_type<ngraph::snippets::op::LoopBegin>(consumer_node) ||
                        ov::is_type<ngraph::op::v0::MatMul>(consumer_node) ||
                        ov::is_type<ngraph::op::v1::Transpose>(consumer_node)) {
                        return false;
                    }
                }
            }

            auto load = std::make_shared<ngraph::snippets::op::Load>(root, count);
            ngraph::copy_runtime_info(root, load);

            bool rewritten = false;
            for (const auto& output : root->outputs()) {
                for (const auto& consumer : output.get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != load) {
                        consumer.replace_source_output(load);
                        rewritten |= true;
                    }
                }
            }

            return rewritten;
        });
}

ngraph::snippets::pass::InsertStore::InsertStore(const size_t count) {
    MATCHER_SCOPE(InsertStore);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::opset1::Result, ngraph::snippets::op::Buffer>(), matcher_name),
            [this, count](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertStore")
            auto root = m.get_match_root();

            // check if already has Store as an input
            for (const auto& input : root->inputs()) {
                const auto& parent_node = input.get_source_output().get_node();
                if (ov::is_type<ngraph::snippets::op::Store>(parent_node) ||
                    ov::is_type<ngraph::snippets::op::LoopEnd>(parent_node) ||
                    ov::is_type<ngraph::op::v0::MatMul>(parent_node)  ||
                    ov::is_type<ngraph::op::v1::Transpose>(parent_node)) {
                    return false;
                }
            }

            auto store = std::make_shared<ngraph::snippets::op::Store>(root->input_value(0), count);
            ngraph::copy_runtime_info(root, store);
            root->set_argument(0, store);
            return true;
        });
}
