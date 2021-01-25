// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::EliminateUnsqueezeGather, "EliminateUnsqueezeGather", 0);

ngraph::pass::EliminateUnsqueezeGather::EliminateUnsqueezeGather() {
    MATCHER_SCOPE(EliminateUnsqueezeGather);
    const auto unsqueezeAxis = ngraph::pattern::any_input();
    const auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset6::Unsqueeze>({ngraph::pattern::any_input(), unsqueezeAxis});
    const auto gatherIndices = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    const auto gatherAxis = ngraph::pattern::any_input();
    const auto gather = ngraph::pattern::wrap_type<ngraph::opset6::Gather>({unsqueeze, gatherIndices, gatherAxis});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& patternValue = m.get_pattern_value_map();

        const auto& m_unsqueezeAxis = patternValue.at(unsqueezeAxis);
        const auto& m_gatherAxis = patternValue.at(gatherAxis);

        const auto& unsqueezeAxisNode = ngraph::as_type_ptr<ngraph::opset6::Constant>(m_unsqueezeAxis.get_node_shared_ptr());
        const auto& gatherAxisNode = ngraph::as_type_ptr<ngraph::opset6::Constant>(m_gatherAxis.get_node_shared_ptr());

        if (!unsqueezeAxisNode || !gatherAxisNode) {
            return false;
        }

        const auto& unsqueezeAxisVec = unsqueezeAxisNode->cast_vector<int64_t>();
        const auto& gatherAxisVec = gatherAxisNode->cast_vector<int64_t>();

        if (unsqueezeAxisVec.size() != 1 || gatherAxisVec.size() != 1) {
            return false;
        }

        if (unsqueezeAxisVec.front() != gatherAxisVec.front()) {
            return false;
        }

        auto& m_gather = patternValue.at(gather);
        const auto& m_unsqueeze = patternValue.at(unsqueeze);
        const auto& unsqueezeData = m_unsqueeze.get_node_shared_ptr()->get_input_node_shared_ptr(0);

        ngraph::copy_runtime_info(m_gather.get_node_shared_ptr(), unsqueezeData);
        m_gather.replace(unsqueezeData);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, "EliminateUnsqueezeGather");
    ngraph::pass::MatcherPass::register_matcher(m, callback);
}
