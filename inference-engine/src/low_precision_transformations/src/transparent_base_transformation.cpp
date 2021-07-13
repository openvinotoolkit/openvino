// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transparent_base_transformation.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

bool TransparentBaseTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    auto operation = m.get_match_root();
    const std::shared_ptr<Node> dequantization = operation->input_value(0).get_node_shared_ptr();
    // const std::shared_ptr<Node> dequantizationParent = dequantization->input_value(0).get_node_shared_ptr();

    // auto newOperation = operation->copy_with_new_inputs({ dequantizationParent });
    // const auto newDequantization = dequantization->copy_with_new_inputs({
    //    newOperation,
    //    dequantization->input_value(1),
    //    dequantization->input_value(2) });

    // const std::string friendlyName = operation->get_friendly_name();
    //// TODO: new operation name has to be unique
    // newOperation->set_friendly_name(friendlyName + "_original");
    // newDequantization->set_friendly_name(friendlyName);

    // replace_node(operation, newDequantization);

    // NetworkHelper::moveDequantization(operation, dequantization);
    return true;
}

bool TransparentBaseTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    return true;
}
