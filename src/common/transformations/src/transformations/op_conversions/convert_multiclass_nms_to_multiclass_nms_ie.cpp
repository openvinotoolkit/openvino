// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <vector>

#include "itt.hpp"
#include "ngraph_ops/nms_static_shape_ie.hpp"

ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE::ConvertMulticlassNmsToMulticlassNmsIE(bool force_i32_output_type) {
    MATCHER_SCOPE(ConvertMulticlassNmsToMulticlassNmsIE);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset9::MulticlassNms>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto nms = std::dynamic_pointer_cast<ngraph::opset9::MulticlassNms>(m.get_match_root());
        if (!nms || transformation_callback(nms)) {
            return false;
        }

        // if input shape is dynamic force the output shape must be dynamic too
        if (nms->get_input_partial_shape(0).is_dynamic() || nms->get_input_partial_shape(1).is_dynamic()) {
            return false;
        }

        const auto new_args = nms->input_values();
        // vector of new nGraph operations
        NodeVector new_ops;
        auto attrs = nms->get_attrs();
        attrs.output_type = force_i32_output_type ? element::i32 : nms->get_output_type();

        auto nms_new = std::make_shared<op::internal::NmsStaticShapeIE<ngraph::opset9::MulticlassNms>>(new_args.at(0),
                                                                                                       new_args.at(1),
                                                                                                       attrs);
        new_ops.emplace_back(nms_new);

        Output<Node> output_0 = nms_new->output(0);
        Output<Node> output_1 = nms_new->output(1);
        Output<Node> output_2 = nms_new->output(2);

        if (nms->output(1).get_element_type() != output_1.get_element_type()) {
            output_1 = std::make_shared<opset1::Convert>(output_1, nms->output(1).get_element_type());
            output_1.get_node_shared_ptr()->set_friendly_name(nms->get_friendly_name() + "/convert.1");
            new_ops.emplace_back(output_1.get_node_shared_ptr());
        }

        if (nms->output(2).get_element_type() != output_2.get_element_type()) {
            output_2 = std::make_shared<opset1::Convert>(output_2, nms->output(2).get_element_type());
            output_2.get_node_shared_ptr()->set_friendly_name(nms->get_friendly_name() + "/convert.2");
            new_ops.emplace_back(output_2.get_node_shared_ptr());
        }

        nms_new->set_friendly_name(nms->get_friendly_name());
        ngraph::copy_runtime_info(nms, new_ops);
        ngraph::replace_node(nms, {output_0, output_1, output_2});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(nms, matcher_name);
    this->register_matcher(m, callback);
}
