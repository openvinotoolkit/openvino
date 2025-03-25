// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fold_convert.hpp"
#include <memory>

#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FoldConvertTransformation::FoldConvertTransformation(const Params& params) : CleanupTransformation(params) {
    MATCHER_SCOPE(FoldConvertTransformation);
    auto subtract = pattern::wrap_type<ov::opset1::Subtract>();
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(subtract, matcher_name);

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    this->register_matcher(matcher, callback);
}

bool FoldConvertTransformation::transform(ov::pass::pattern::Matcher &m) {
    const auto subtract = m.get_match_root();
    if (!canBeTransformed(subtract)) {
        return false;
    }

    auto foldConvert = [&](const size_t branch) {
        const auto convert = subtract->get_input_node_shared_ptr(branch);
        if (!ov::is_type<ov::opset1::Convert>(convert) || !ov::is_type<ov::opset1::Constant>(convert->get_input_node_shared_ptr(0))) {
            return;
        }

        const auto resultConstant = ov::pass::low_precision::foldConvert(convert->input_value(0), convert->get_output_element_type(0));
        assert(ov::is_type<ov::opset1::Constant>(resultConstant));

        replace_node(convert, resultConstant);
        updateOutput(resultConstant, convert);
    };

    foldConvert(0ul);
    foldConvert(1ul);

    return true;
}

bool FoldConvertTransformation::canBeTransformed(const std::shared_ptr<Node>& operation) const {
    return
        CleanupTransformation::canBeTransformed(operation) &&
        ((ov::is_type<ov::opset1::Convert>(operation->get_input_node_ptr(1)) &&
        ov::is_type<ov::opset1::Constant>(operation->get_input_node_ptr(1)->get_input_node_ptr(0))) ||
        (ov::is_type<ov::opset1::Convert>(operation->get_input_node_ptr(0)) &&
        ov::is_type<ov::opset1::Constant>(operation->get_input_node_ptr(0)->get_input_node_ptr(0))));
}

bool FoldConvertTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
