// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_nonzero.hpp"

#include <vpu/ngraph/operations/static_shape_nonzero.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

#include <ngraph/opsets/opset3.hpp>

#include <memory>

namespace ngraph {
namespace pass {

DynamicToStaticShapeNonZero::DynamicToStaticShapeNonZero() {
    // We don't set strict_mode when use pattern Matcher,
    // so we can set any type and shape for input.
    auto inputWithAnyTypeAndShape = std::make_shared<pattern::op::Label>(
            element::dynamic, PartialShape{});
    auto nonZeroPattern = std::make_shared<ngraph::op::NonZero>(inputWithAnyTypeAndShape);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& matcher) {
        const auto nonZero = std::dynamic_pointer_cast<ngraph::opset3::NonZero>(matcher.get_match_root());
        if (!nonZero) {
            return false;
        }

        auto staticShapeNonZero = std::make_shared<ngraph::op::StaticShapeNonZero>(
                nonZero->input(0).get_source_output());
        staticShapeNonZero->set_friendly_name(nonZero->get_friendly_name() + "/static_shape");

        auto dynamicShapeResolver = std::make_shared<ngraph::op::DynamicShapeResolver>(
                staticShapeNonZero->output(0), staticShapeNonZero->output(1));
        dynamicShapeResolver->set_friendly_name(nonZero->get_friendly_name() + "/resolve_shape");

        ngraph::replace_node(matcher.get_match_root(), dynamicShapeResolver);
        return true;
    };

    const auto matcher = std::make_shared<ngraph::pattern::Matcher>(
            nonZeroPattern, "DynamicToStaticShapeNonZero");
    this->add_matcher(matcher, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

}  // namespace pass
}  // namespace ngraph
