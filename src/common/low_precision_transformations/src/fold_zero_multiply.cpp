// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fold_zero_multiply.hpp"

#include <memory>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <string>
#include <vector>

#include "itt.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FoldZeroMultiplyTransformation::FoldZeroMultiplyTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FoldZeroMultiplyTransformation);
    auto matcher = pattern::wrap_type<opset1::Multiply>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

//    y = multiply(x, [0,0,....,0])
// => y = Broadcast(Constant(0, typeof(y))), ShapeOf(x))
bool FoldZeroMultiplyTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    const auto multiplyNode = ov::as_type_ptr<opset1::Multiply>(m.get_match_root());
    if (!multiplyNode) {
        return false;
    }

    if (!canBeTransformed(context, multiplyNode)) {
        return false;
    }

    auto dataNode = multiplyNode->get_input_node_shared_ptr(0);
    if (ov::as_type_ptr<opset1::Constant>(dataNode)) {
        dataNode = multiplyNode->get_input_node_shared_ptr(1);
    }

    const auto data_shape_node = fold<opset1::ShapeOf>(dataNode);
    const auto zero_const =
        std::make_shared<opset1::Constant>(multiplyNode->get_output_element_type(0), Shape{}, std::vector<size_t>({0}));
    auto result = fold<opset1::Broadcast>(zero_const, data_shape_node);
    replace_node(multiplyNode, result);
    return true;
}

bool FoldZeroMultiplyTransformation::canBeTransformed(const TransformationContext& context,
                                                      std::shared_ptr<Node> op) const {
    auto constInput = NetworkHelper::getConstantInput(op);
    if (!constInput) {
        return false;
    }

    auto constNode = ov::as_type_ptr<opset1::Constant>(constInput);
    if (!constNode) {
        return false;
    }

    const auto scales = constNode->cast_vector<float>();
    return std::all_of(scales.begin(), scales.end(), [](const float& v) {
        return v == 0.0f;
    });
}

bool FoldZeroMultiplyTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
