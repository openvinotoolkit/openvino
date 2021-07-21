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

#include "ngraph_ops/nms_static_shape_ie.hpp"
#include "transformations/op_conversions/convert_matrix_nms_to_matrix_nms_ie.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertMatrixNmsToMatrixNmsIE, "ConvertMatrixNmsToMatrixNmsIE", 0);

ngraph::pass::ConvertMatrixNmsToMatrixNmsIE::ConvertMatrixNmsToMatrixNmsIE() {
    MATCHER_SCOPE(ConvertMatrixNmsToMatrixNmsIE);
    auto nms = ngraph::pattern::wrap_type<ngraph::opset8::MatrixNms>();

    ngraph::matcher_pass_callback callback = [](pattern::Matcher &m) {
        auto nms = std::dynamic_pointer_cast<ngraph::opset8::MatrixNms>(m.get_match_root());
        if (!nms) {
            return false;
        }

        const auto new_args = nms->input_values();
        // vector of new nGraph operations
        NodeVector new_ops;
        auto attrs = nms->get_attrs();
        attrs.output_type = element::i32;
        auto nms_new = std::make_shared<op::internal::NmsStaticShapeIE<ngraph::opset8::MatrixNms>>(
                new_args.at(0),
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
