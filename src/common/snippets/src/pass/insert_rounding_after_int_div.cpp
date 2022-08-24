// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_rounding_after_int_div.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::InsertRoundingAfterIntDiv::InsertRoundingAfterIntDiv() {
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
            if (!div)
                return false;

            const bool is_pythondiv = div->is_pythondiv();

            // check if already has rounding op as an output of divide op
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if ((is_pythondiv && ov::is_type<ov::op::v0::Floor>(consumer.get_node())) ||
                        (!is_pythondiv && ov::is_type<ngraph::snippets::op::Truncation>(consumer.get_node()))) {
                        return false;
                    }
                }
            }

            std::shared_ptr<ov::Node> rounding = nullptr;
            if (is_pythondiv) {
                rounding = std::make_shared<ov::op::v0::Floor>(div);
            } else {
                rounding = std::make_shared<ngraph::snippets::op::Truncation>(div);
            }
            ngraph::copy_runtime_info(root, rounding);

            bool rewritten = false;
            for (auto output : root->outputs()) {
                for (auto consumer : output.get_target_inputs()) {
                    if (consumer.get_node()->shared_from_this() != rounding) {
                        consumer.replace_source_output(rounding);
                        rewritten |= true;
                    }
                }
            }

            return rewritten;
        });
}
