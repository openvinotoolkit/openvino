// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include "snippets/snippets_isa.hpp"
#include "snippets/pass/convert_constants.hpp"
#include <ngraph/rt_info.hpp>


bool ngraph::snippets::pass::ConvertConstants::run_on_model(const std::shared_ptr<ov::Model>& m) {
    MATCHER_SCOPE(ConvertConstants);

    bool was_updated = false;
    ngraph::ParameterVector parameters;

    const auto& ops = m->get_ops();
    for (auto& op : ops) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertConstants")

        auto constant = as_type_ptr<ov::op::v0::Constant>(op);
        if (constant == nullptr) {
            continue;
        }

        std::shared_ptr<Node> replacement;
        if (ngraph::shape_size(constant->get_shape()) == 1ul) {
            replacement = std::make_shared<snippets::op::Scalar>(*constant);
        } else {
            auto parameter = std::make_shared<opset1::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());
            parameters.push_back(parameter);
            replacement = parameter;
        }

        replacement->set_friendly_name(constant->get_friendly_name());
        ngraph::copy_runtime_info(constant, replacement);
        ngraph::replace_node(constant, replacement);

        was_updated = true;
    }
    m->add_parameters(parameters);

    return was_updated;
}
