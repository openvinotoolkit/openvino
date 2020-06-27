// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/add.hpp"

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

void AddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::Add>(
                    { make_op_label<opset1::Multiply>(),
                      make_op_label<opset1::Constant>()}));
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::Add>(
                    { make_op_label<opset1::Constant>(),
                      make_op_label<opset1::Multiply>() }));
}

void AddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto add = as_type_ptr<opset1::Add>(m.get_match_root());
    if (add->get_friendly_name() == "InceptionV2/InceptionV2/Conv2d_2c_3x3/BatchNorm/FusedBatchNormV3/variance/Fused_Add_") {
        std::cout << "AddTransformation::transform: " << add->get_friendly_name() << std::endl;
    }
    // Limited implementation: fuse only Add(Multiply, Const) or Add(Const, Multply), nothing else

    // Figure out where SS and where is Constant
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(add->input_value(0).get_node_shared_ptr());
    std::shared_ptr<opset1::Multiply> multiply;
    if (constant) {
        multiply = as_type_ptr<opset1::Multiply>(add->input_value(1).get_node_shared_ptr());
        assert(false);  // requirement from out swapper of Multiply and Add
    } else {
        constant = as_type_ptr<opset1::Constant>(add->input_value(1).get_node_shared_ptr());
        multiply = as_type_ptr<opset1::Multiply>(add->input_value(0).get_node_shared_ptr());
    }

    assert(constant && multiply);

    // TODO: FIXME: output names
    auto newMultiply = swapMultiplyAndAdd(add);
    newMultiply->set_friendly_name(add->get_friendly_name());
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
