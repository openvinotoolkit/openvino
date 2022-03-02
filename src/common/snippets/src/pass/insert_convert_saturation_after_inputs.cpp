// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/remarks.hpp"

#include "snippets/pass/insert_convert_saturation_after_inputs.hpp"
#include "snippets/snippets_isa.hpp"

#include "ngraph/type.hpp"
#include "ngraph/node.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>


bool insertConvertSaturationAfterNode(const std::shared_ptr<ngraph::Node>& node, const ov::element::Type element_type) {
    bool rewritten = false;
    for (const auto& output : node->outputs()) {
        for (auto consumer : output.get_target_inputs()) {
            // If after node there is ConvertTruncation we should insert ConvertSaturation after that
            if (auto existing_convert_t = ngraph::as_type_ptr<ngraph::snippets::op::ConvertTruncation>(consumer.get_node()->shared_from_this())) {
                if (existing_convert_t->get_destination_type() != element_type) {
                    rewritten = insertConvertSaturationAfterNode(existing_convert_t, element_type);
                }
                continue;
            }

            auto existing_convert_s = ngraph::as_type_ptr<ngraph::snippets::op::ConvertSaturation>(consumer.get_node()->shared_from_this());
            if ((!existing_convert_s && !ov::is_type<ngraph::op::v0::Result>(consumer.get_node()->shared_from_this()) &&
                    consumer.get_element_type() != element_type) ||
                (existing_convert_s && existing_convert_s->get_destination_type() != element_type)) {
                const auto convert = std::make_shared<ngraph::snippets::op::ConvertSaturation>(node, element_type);
                consumer.replace_source_output(convert);
                rewritten |= true;
            }
        }
    }
    return rewritten;
}

ngraph::snippets::pass::InsertConvertSaturationAfterInputs::InsertConvertSaturationAfterInputs(const ov::element::Type exec_type) {
    MATCHER_SCOPE(InsertConvertSaturationAfterInputs);

    auto param_pattern = ngraph::pattern::wrap_type<ngraph::opset1::Parameter>();
    auto scalar_pattern = pattern::wrap_type<opset1::Constant>(
        [=](Output<Node> output) -> bool { return ngraph::shape_size(output.get_shape()) == 1; });
    auto input = std::make_shared<pattern::op::Or>(OutputVector{ param_pattern, scalar_pattern });

    ngraph::matcher_pass_callback callback = [this, exec_type](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::InsertConvertSaturationAfterInputs")
        auto root = m.get_match_root();

        auto rewritten = insertConvertSaturationAfterNode(root, exec_type);

        return rewritten;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(input, matcher_name);
    register_matcher(m, callback);
}
