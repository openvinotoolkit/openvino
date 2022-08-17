// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_floor_after_int_div.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::InsertFloorAfterIntDiv::InsertFloorAfterIntDiv() {
    MATCHER_SCOPE(InsertFloorAfterIntDiv);
    auto divide = std::make_shared<pattern::op::Label>(pattern::any_input(),
                                                    [](std::shared_ptr<Node> n) {
                                                        return is_type<ov::op::v1::Divide>(n) &&
                                                               n->get_input_element_type(0).is_integral_number() &&
                                                               n->get_input_element_type(1).is_integral_number() &&
                                                               n->get_output_element_type(0).is_integral_number();
                                                    });

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(divide, matcher_name),
        [this](ngraph::pattern::Matcher &m) {
            OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertFloorAfterIntDiv")
            auto root = m.get_match_root();
            auto div = ov::as_type_ptr<ov::op::v1::Divide>(root);
            if (!div || !div->is_pythondiv())
                return false;

            // check if already has Floor as an output
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (ov::is_type<ngraph::snippets::op::Load>(consumer.get_node())) {
                        return false;
                    }
                }
            }

            auto floor = std::make_shared<ov::op::v0::Floor>(div);
            ngraph::copy_runtime_info(root, floor);

            bool rewritten = false;
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != floor) {
                        consumer.replace_source_output(floor);
                        rewritten |= true;
                    }
                }
            }

            return rewritten;
        });
}
