// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/convert.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ConvertTransformation, "ConvertTransformation", 0);

ConvertTransformation::ConvertTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::Convert>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ConvertTransformation");
    this->register_matcher(m, callback);
}

bool ConvertTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) {
    std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(m.get_match_root());
    if (!convert) {
        return false;
    }

    if (!canBeTransformed(context, convert)) {
        return false;
    }

    const ngraph::element::Type precisionBefore = convert->get_input_element_type(0);

    std::shared_ptr<opset1::Subtract> subtract = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
        convert->get_input_node_shared_ptr(0),
        std::make_shared<opset1::Constant>(precisionBefore, Shape{}, std::vector<size_t>({ 0 })));
    NetworkHelper::setOutDataPrecision(subtract, convert->get_output_element_type(0));

    replace_node(convert, subtract);

    subtract->set_friendly_name(convert->get_friendly_name());
    return true;
}

bool ConvertTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
