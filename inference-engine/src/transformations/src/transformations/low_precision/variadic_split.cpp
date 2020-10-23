// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/variadic_split.hpp"
#include "ngraph/node.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {
VariadicSplitTransformation::VariadicSplitTransformation(const Params& params) : SplitTransformation(params) {}

void VariadicSplitTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::VariadicSplit>({
                    make_op_label<opset1::Multiply>(),
                    make_op_label<opset1::Constant>(),
                    make_op_label<opset1::Constant>() }));
}

std::vector<size_t> VariadicSplitTransformation::getConstSplitLengths(
    const OutputVector& inputs,
    const ngraph::Shape& constShape,
    const size_t outputSize) const {
    std::vector<size_t> lengths = as_type_ptr<opset1::Constant>(inputs[2].get_node_shared_ptr())->cast_vector<size_t>();

    int64_t axis = as_type_ptr<opset1::Constant>(inputs[1].get_node_shared_ptr())->cast_vector<int64_t>()[0];
    size_t splitedAxis = axis > 0 ? axis : inputs[0].get_shape().size() + axis;

    if ((!constShape.empty()) && (constShape[splitedAxis] != 1)) {
        std::vector<size_t> result(outputSize + 1);
        result[0] = 0;
        for (size_t i = 1; i < result.size(); ++i) {
            result[i] = result[i - 1] + lengths[i - 1];
        }
        return result;
    } else {
        return std::vector<size_t>();
    }
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
