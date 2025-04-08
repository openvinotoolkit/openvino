// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/eliminate_unsqueeze_gather.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_bitwise.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::op::util;
using namespace ov::pass::pattern;

ov::pass::EliminateUnsqueezeGather::EliminateUnsqueezeGather() {
    MATCHER_SCOPE(EliminateUnsqueezeGather);
    // Remove Unsqueeze + Gather pair, if Gather gathers data by `1` dimension that was previously added by Unsqueeze
    const auto unsqueezeAxis = any_input();
    const auto unsqueezeInput = any_input();
    const auto unsqueeze = wrap_type<v0::Unsqueeze>({unsqueezeInput, unsqueezeAxis}, consumers_count(1));
    const auto gatherIndices = v0::Constant::create(element::i64, Shape{}, {0});
    const auto gatherAxis = any_input();
    const auto gather = wrap_type<GatherBase>({unsqueeze, gatherIndices, gatherAxis});

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

inline bool scalar_with_one_consumer(const Output<Node>& out) {
    return rank_equals(0)(out) && consumers_count(1)(out);
}

ov::pass::EliminateGatherUnsqueeze::EliminateGatherUnsqueeze() {
    MATCHER_SCOPE(EliminateGatherUnsqueeze);

    const auto gather_label = wrap_type<GatherBase>(scalar_with_one_consumer);
    const auto be_label = wrap_type<BinaryElementwiseArithmetic, BinaryElementwiseComparison, BinaryElementwiseLogical>(
        {gather_label, any_input()},
        scalar_with_one_consumer);
    const auto or_label = std::make_shared<pattern::op::Or>(OutputVector{gather_label, be_label});
    const auto unsqueeze_label = wrap_type<v0::Unsqueeze, v1::Reshape>({or_label, any_input()}, rank_equals(1));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto pattern_nodes = m.get_pattern_map();
        auto& gather = pattern_nodes.at(gather_label);
        auto& unsqueeze = pattern_nodes.at(unsqueeze_label);
        const auto& indices = op::util::make_try_fold<v1::Reshape>(gather->input_value(1),
                                                                   v0::Constant::create(element::i32, {1}, {1}),
                                                                   false);
        register_new_node(indices);
        gather->input(1).replace_source_output(indices->output(0));
        copy_runtime_info({unsqueeze, gather}, {indices, gather});
        replace_output_update_name(unsqueeze->output(0), unsqueeze->input_value(0));

        // in order to have correct shapes for other matchers in the same graph rewrite we revalidate nodes
        gather->revalidate_and_infer_types();
        if (pattern_nodes.count(be_label))
            pattern_nodes.at(be_label)->revalidate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<Matcher>(unsqueeze_label, matcher_name);
    register_matcher(m, callback);
}
