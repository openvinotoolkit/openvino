// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/fuse_load_and_convert.hpp"

#include <snippets/itt.hpp>
#include "remarks.hpp"

#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

ngraph::snippets::pass::FuseLoadAndConvert::FuseLoadAndConvert(const std::vector<element::Type> supported_types) {
    MATCHER_SCOPE(FuseLoadAndConvert);

    // TODO: add supported_types in matcher
    const auto matcher = std::make_shared<ngraph::pattern::Matcher>(ngraph::pattern::wrap_type<ngraph::opset1::Convert>(
        {ngraph::pattern::wrap_type<ngraph::snippets::op::Load>()}));

    const auto callback = [this, supported_types](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::FuseLoadAndConvert")
        auto convert = m.get_match_root();
        if (transformation_callback(convert)) {
            return false;
        }

        const auto actual_type = convert->get_output_element_type(0);
        if (std::all_of(
            supported_types.begin(),
            supported_types.end(),
            [actual_type](const element::Type supported_type) { return supported_type != actual_type; })) {
            // no possible to fuse: Convert output type is not supported
            return false;
        }

        auto load = convert->get_input_node_shared_ptr(0);

        auto newLoad = std::make_shared<ngraph::op::TypeRelaxed<ngraph::snippets::op::Load>>(
            *ngraph::as_type_ptr<ngraph::snippets::op::Load>(load->clone_with_new_inputs({load->get_input_node_shared_ptr(0)})),
            convert->get_output_element_type(0));
        ;
        newLoad->set_output_type(0, convert->get_output_element_type(0), newLoad->output(0).get_partial_shape());
        replace_node(convert, newLoad);
        ngraph::copy_runtime_info({load, convert}, newLoad);

        return true;
    };

    register_matcher(matcher, callback);
}
