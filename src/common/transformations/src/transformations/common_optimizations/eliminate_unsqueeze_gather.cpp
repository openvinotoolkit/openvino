// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"

ngraph::pass::EliminateUnsqueezeGather::EliminateUnsqueezeGather() {
    MATCHER_SCOPE(EliminateUnsqueezeGather);
    // Remove Unsqueeze + Gather pair, if Gather gathers data by `1` dimension that was previously added by Unsqueeze
    const auto unsqueezeAxis = ngraph::pattern::any_input();
    const auto unsqueezeInput = ngraph::pattern::any_input();
    const auto unsqueeze = ngraph::pattern::wrap_type<ngraph::opset6::Unsqueeze>({unsqueezeInput, unsqueezeAxis},
                                                                                 pattern::consumers_count(1));
    const auto gatherIndices = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    const auto gatherAxis = ngraph::pattern::any_input();
    const auto gather =
        ngraph::pattern::wrap_type<ngraph::op::util::GatherBase>({unsqueeze, gatherIndices, gatherAxis});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& patternValue = m.get_pattern_value_map();

        const auto& m_unsqueezeAxis = patternValue.at(unsqueezeAxis);
        const auto& m_gatherAxis = patternValue.at(gatherAxis);

        const auto& unsqueezeAxisNode =
            ngraph::as_type_ptr<ngraph::opset6::Constant>(m_unsqueezeAxis.get_node_shared_ptr());
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
        const auto& m_unsqueezeInput = patternValue.at(unsqueezeInput);

        ngraph::copy_runtime_info(m_gather.get_node_shared_ptr(), m_unsqueeze.get_node_shared_ptr());
        m_gather.replace(m_unsqueezeInput);
        MATCHER_SCOPE_ENABLE(EliminateUnsqueezeGather);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gather, "EliminateUnsqueezeGather");
    register_matcher(m, callback);
}

ngraph::pass::EliminateGatherUnsqueeze::EliminateGatherUnsqueeze() {
    MATCHER_SCOPE(EliminateGatherUnsqueeze);

    const auto gather_indices_label = ngraph::pattern::wrap_type<ngraph::op::Constant>(pattern::rank_equals(0));
    const auto gather_axis_label = ngraph::pattern::wrap_type<ngraph::op::Constant>();
    const auto gather_label = ngraph::pattern::wrap_type<ngraph::op::util::GatherBase>(
        {ngraph::pattern::any_input(), gather_indices_label, gather_axis_label},
        pattern::rank_equals(0));

    const auto unsqueeze_label =
        ngraph::pattern::wrap_type<ngraph::opset6::Unsqueeze>({gather_label, ngraph::pattern::any_input()},
                                                              pattern::rank_equals(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto pattern_nodes = m.get_pattern_map();

        auto& gather_indices = pattern_nodes.at(gather_indices_label);
        auto& gather = pattern_nodes.at(gather_label);
        auto& unsqueeze = pattern_nodes.at(unsqueeze_label);

        auto new_indices =
            ngraph::op::util::make_try_fold<ngraph::opset6::Reshape>(gather_indices,
                                                                     opset6::Constant::create(element::i32, {1}, {1}),
                                                                     false);
        auto new_gather = gather->clone_with_new_inputs({gather->input_value(0), new_indices, gather->input_value(2)});

        new_gather->set_friendly_name(gather->get_friendly_name());
        ngraph::copy_runtime_info({unsqueeze, gather}, {new_gather, new_indices});
        ngraph::replace_node(unsqueeze, new_gather);
        MATCHER_SCOPE_ENABLE(EliminateGatherUnsqueeze);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(unsqueeze_label, "EliminateGatherUnsqueeze");
    register_matcher(m, callback);
}
