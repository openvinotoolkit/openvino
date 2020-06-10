// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/low_precision/transparent_base_transformation.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ngraph_ops/multiply_add.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

void TransparentBaseTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto operation = m.get_match_root();
    const std::shared_ptr<Node> dequantization = operation->input_value(0).get_node_shared_ptr();
    const std::shared_ptr<Node> dequantizationParent = dequantization->input_value(0).get_node_shared_ptr();

    auto newOperation = operation->copy_with_new_inputs({ dequantizationParent });
    const auto newDequantization = dequantization->copy_with_new_inputs({
        newOperation,
        dequantization->input_value(1),
        dequantization->input_value(2) });

    const std::string friendlyName = operation->get_friendly_name();
    // TODO: new operation name has to be equal
    newOperation->set_friendly_name(friendlyName + "_original");
    newDequantization->set_friendly_name(friendlyName);

    replace_node(operation, newDequantization);
}

bool TransparentBaseTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    const std::shared_ptr<ngraph::op::MultiplyAdd> dequantization = as_type_ptr<ngraph::op::MultiplyAdd>(layer->input_value(0).get_node_shared_ptr());
    return dequantization != nullptr;
}
