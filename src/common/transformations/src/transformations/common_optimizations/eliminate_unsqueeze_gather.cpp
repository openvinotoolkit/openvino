// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass::pattern;

ov::pass::EliminateUnsqueezeGather::EliminateUnsqueezeGather() {
    MATCHER_SCOPE(EliminateUnsqueezeGather);
    // Remove Unsqueeze + Gather pair, if Gather gathers data by `1` dimension that was previously added by Unsqueeze
    const auto unsqueezeAxis = any_input();
    const auto unsqueezeInput = any_input();
    const auto unsqueeze = wrap_type<v0::Unsqueeze>({unsqueezeInput, unsqueezeAxis}, consumers_count(1));
    const auto gatherIndices = v0::Constant::create(element::i64, Shape{}, {0});
    const auto gatherAxis = any_input();
    const auto gather = wrap_type<op::util::GatherBase>({unsqueeze, gatherIndices, gatherAxis});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto& patternValue = m.get_pattern_value_map();
        const auto& m_unsqueezeAxis = patternValue.at(unsqueezeAxis);
        const auto& m_gatherAxis = patternValue.at(gatherAxis);
        const auto& unsqueezeAxisNode = as_type_ptr<v0::Constant>(m_unsqueezeAxis.get_node_shared_ptr());
        const auto& gatherAxisNode = as_type_ptr<v0::Constant>(m_gatherAxis.get_node_shared_ptr());

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

        copy_runtime_info(m_gather.get_node_shared_ptr(), m_unsqueeze.get_node_shared_ptr());
        m_gather.replace(m_unsqueezeInput);

        return true;
    };

    auto m = std::make_shared<Matcher>(gather, matcher_name);
    register_matcher(m, callback);
}

ov::pass::EliminateGatherUnsqueeze::EliminateGatherUnsqueeze() {
    MATCHER_SCOPE(EliminateGatherUnsqueeze);

    const auto indices_label = wrap_type<v0::Constant>(rank_equals(0));
    const auto axis_label = wrap_type<v0::Constant>();
    const auto gather_label = wrap_type<op::util::GatherBase>({any_input(), indices_label, axis_label}, rank_equals(0));

    const auto unsqueeze_label = wrap_type<v0::Unsqueeze, v1::Reshape>({gather_label, any_input()}, rank_equals(1));

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        auto pattern_nodes = m.get_pattern_map();

        auto& gather_indices = pattern_nodes.at(indices_label);
        auto& gather = pattern_nodes.at(gather_label);
        auto& unsqueeze = pattern_nodes.at(unsqueeze_label);

        auto new_indices =
            op::util::make_try_fold<::v1::Reshape>(gather_indices, v0::Constant::create(element::i32, {1}, {1}), false);
        auto new_gather = gather->clone_with_new_inputs({gather->input_value(0), new_indices, gather->input_value(2)});

        new_gather->set_friendly_name(gather->get_friendly_name());
        copy_runtime_info({unsqueeze, gather}, {new_gather, new_indices});
        replace_node(unsqueeze, new_gather);
        return true;
    };

    auto m = std::make_shared<Matcher>(unsqueeze_label, matcher_name);
    register_matcher(m, callback);
}
