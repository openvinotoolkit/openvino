// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strip_noop_gather_ids_reshape.hpp"

#include <memory>

#include "../partitioning/patterns/opt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

class StripNoopGatherIdsReshapeMatcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::StripNoopGatherIdsReshapeMatcher");

    StripNoopGatherIdsReshapeMatcher() {
        auto vocab = opp::any_input([](const ov::Output<ov::Node>& output) {
            return output.get_partial_shape().is_static() && output.get_shape().size() == 2 &&
                   output.get_shape()[1] >= ov::npuw::patterns::opt::MIN_HOST_GATHER_HIDDEN_SIZE;
        });
        auto source = opp::any_input(opp::has_static_shape());
        auto reshape = opp::wrap_type<ov::op::v1::Reshape>({source, opp::any_input()});
        auto ids = opp::wrap_type<ov::op::v0::Squeeze>({reshape, opp::any_input()}, opp::has_static_shape());
        auto gather = opp::wrap_type<ov::op::v8::Gather>({vocab, ids, opp::any_input()});

        ov::matcher_pass_callback callback = [=](opp::Matcher& matcher) {
            const auto& pattern_map = matcher.get_pattern_value_map();
            const auto& source_output = pattern_map.at(source);
            if (source_output.get_shape() != pattern_map.at(ids).get_shape()) {
                return false;
            }
            pattern_map.at(gather).get_node()->input(1).replace_source_output(source_output);
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(gather, "StripNoopGatherIdsReshapeMatcher"), callback);
    }
};

ov::npuw::StripNoopGatherIdsReshape::StripNoopGatherIdsReshape() {
    add_matcher<StripNoopGatherIdsReshapeMatcher>();
}
