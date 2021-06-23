// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset8.hpp>

#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "ngraph_ops/multiclass_nms_ie_internal.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie_internal.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIEInternal, "ConvertMulticlassNmsToMulticlassNmsIEInternal", 0);

ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIEInternal::ConvertMulticlassNmsToMulticlassNmsIEInternal() {
    MATCHER_SCOPE(ConvertMulticlassNmsToMulticlassNmsIEInternal);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset8::MulticlassNms>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms = std::dynamic_pointer_cast<ngraph::opset8::MulticlassNms>(m.get_match_root());
        if (!nms) {
            return false;
        }

        const auto new_args = nms->input_values();
        // vector of new nGraph operations
        NodeVector new_ops;

        int sort_result_type = 0;
        switch (nms->get_sort_result_type()) {
            case ::ngraph::opset8::MulticlassNms::SortResultType::SCORE:
                sort_result_type = 1;
                break;
            case ::ngraph::opset8::MulticlassNms::SortResultType::CLASSID:
                sort_result_type = 0;
                break;
            case ::ngraph::opset8::MulticlassNms::SortResultType::NONE:
                sort_result_type = 2;
                break;
            default:
                throw ngraph_error("MulticlassNms layer " + nms->get_friendly_name() +
                                   " has unsupported sort result type");
        }

        std::shared_ptr<op::internal::MulticlassNmsIEInternal> nms_legacy{nullptr};

        nms_legacy = std::make_shared<op::internal::MulticlassNmsIEInternal>(
                new_args.at(0),
                new_args.at(1),
                sort_result_type,
                nms->get_sort_result_across_batch(),
                element::i32,
                nms->get_iou_threshold(),
                nms->get_score_threshold(),
                nms->get_nms_top_k(),
                nms->get_keep_top_k(),
                nms->get_background_class(),
                nms->get_nms_eta(),
                nms->get_normalized());
        new_ops.emplace_back(nms_legacy);

        Output<Node> output_0 = nms_legacy->output(0);

        Output<Node> output_1 = nms_legacy->output(1);
        if (nms->output(1).get_element_type() != output_1.get_element_type()) {
            output_1 = std::make_shared<opset1::Convert>(output_1, nms->output(1).get_element_type());
            output_1.get_node_shared_ptr()->set_friendly_name(nms->get_friendly_name() + "/convert.1");
            new_ops.emplace_back(output_1.get_node_shared_ptr());
        }

        Output<Node> output_2 = nms_legacy->output(2);
        if (nms->output(2).get_element_type() != output_2.get_element_type()) {
            output_2 = std::make_shared<opset1::Convert>(output_2, nms->output(2).get_element_type());
            output_2.get_node_shared_ptr()->set_friendly_name(nms->get_friendly_name() + "/convert.2");
            new_ops.emplace_back(output_2.get_node_shared_ptr());
        }

        nms_legacy->set_friendly_name(nms->get_friendly_name());
        ngraph::copy_runtime_info(nms, new_ops);
        ngraph::replace_node(nms, {output_0, output_1, output_2});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}
