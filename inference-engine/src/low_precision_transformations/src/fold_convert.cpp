// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fold_convert.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/fake_quantize.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FoldConvertTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Subtract>(pass, context);
}

bool FoldConvertTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    const auto subtract = m.get_match_root();
    if (!canBeTransformed(context, subtract)) {
        return false;
    }

    auto foldConvert = [&](const size_t branch) {
        const auto convert = subtract->get_input_node_shared_ptr(branch);
        if (!is_type<opset1::Convert>(convert) || !is_type<opset1::Constant>(convert->get_input_node_shared_ptr(0))) {
            return;
        }

        const auto resultConstant = ngraph::pass::low_precision::foldConvert(convert->get_input_node_shared_ptr(0), convert->output(0).get_element_type());
        assert(is_type<opset1::Constant>(resultConstant));

        replace_node(convert, resultConstant);
        updateOutput(context, resultConstant, convert);
    };

    foldConvert(0ul);
    foldConvert(1ul);

    return true;
}

bool FoldConvertTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    return
        (is_type<opset1::Convert>(operation->get_input_node_ptr(1)) &&
        is_type<opset1::Constant>(operation->get_input_node_ptr(1)->get_input_node_ptr(0))) ||
        (is_type<opset1::Convert>(operation->get_input_node_ptr(0)) &&
        is_type<opset1::Constant>(operation->get_input_node_ptr(0)->get_input_node_ptr(0)));
}

bool FoldConvertTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
