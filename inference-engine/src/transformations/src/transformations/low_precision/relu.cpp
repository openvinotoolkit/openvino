// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/relu.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"


namespace ngraph {
namespace pass {
namespace low_precision {

void ReluTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::Relu>(
                    { make_op_label<opset1::Multiply>()}));
}

void ReluTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto relu = as_type_ptr<opset1::Relu>(m.get_match_root());
    auto multiply = as_type_ptr<opset1::Multiply>(relu->input_value(0).get_node_shared_ptr());

    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(multiply->input_value(0).get_node_shared_ptr());
    auto data = multiply->input_value(1);
    if (!constant) {
        constant = as_type_ptr<opset1::Constant>(multiply->input_value(1).get_node_shared_ptr());
        data = multiply->input_value(0);
    }

    assert(constant);

    // Check if all scales are greater than zero
    // TODO: supply necessary references in ngraph to write something like this
    // fold<opset1::ReduceLogicalAnd>(fold<opset1::GreaterEqual>(constant,
    //  std::make_shared<opset1::Constant>(constant->get_output_element_type(0), Shape{}, 0)));

    auto scales = constant->cast_vector<float>();
    if (std::all_of(scales.begin(), scales.end(), [](float value) {
        return value >= 0.0;
    })) {
        auto replacement = multiply->copy_with_new_inputs({relu->copy_with_new_inputs({data}), constant});
        replace_node(relu, replacement);
        replacement->set_friendly_name(relu->get_friendly_name());
        replacement->input_value(0).get_node_shared_ptr()->set_friendly_name(replacement->input_value(0).get_node_shared_ptr()->get_name());
    }

    // std::cout << "ReluTransformation::transform: done: " << relu->get_friendly_name() << std::endl;
}

}// namespace low_precision
}// namespace pass
}// namespace ngraph
