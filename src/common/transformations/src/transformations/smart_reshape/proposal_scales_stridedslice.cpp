// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/proposal_scales_stridedslice.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/proposal.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace {

bool crop_scales_for_proposal(const ov::pass::pattern::PatternValueMap& pattern_to_output,
                              const std::shared_ptr<ov::Node>& parameter_label,
                              const std::shared_ptr<ov::Node>& proposal_label) {
    const auto& parameter = pattern_to_output.at(parameter_label);
    const auto& proposal = pattern_to_output.at(proposal_label).get_node_shared_ptr();

    auto cropped_scales = std::make_shared<ov::op::v1::StridedSlice>(
        proposal->input_value(2),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {parameter.get_partial_shape()[1].get_length()}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
        std::vector<int64_t>{0},
        std::vector<int64_t>{0});

    proposal->input(2).replace_source_output(cropped_scales->output(0));
    return true;
}

}  // namespace

ov::pass::Proposal1Scales::Proposal1Scales() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(Proposal1Scales);
    auto parameter_label = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>([](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() &&
               (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });

    auto optional_convert = pattern::optional<ov::op::v0::Convert>(parameter_label);
    auto reshape_label = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>(
        {optional_convert, ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });
    auto proposal_label =
        ov::pass::pattern::wrap_type<ov::op::v0::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher& m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(proposal_label /*, matcher_name */);
    register_matcher(m, callback);
}

ov::pass::Proposal4Scales::Proposal4Scales() {
    // TODO: enable conditional compile
    // MATCHER_SCOPE(Proposal4Scales);
    auto parameter_label = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>([](const Output<Node>& output) {
        const auto& shape = output.get_partial_shape();
        return shape.rank().is_static() && shape.rank().get_length() == 2 && shape[1].is_static() &&
               (shape[1].get_length() == 3 || shape[1].get_length() == 4);
    });
    auto optional_convert = ov::pass::pattern::optional<ov::op::v0::Convert>(parameter_label);
    auto reshape_label = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>(
        {optional_convert, ov::pass::pattern::wrap_type<ov::op::v0::Constant>()},
        [](const Output<Node>& output) {
            return output.get_partial_shape().rank().is_static() && output.get_partial_shape().rank().get_length() == 1;
        });
    auto proposal_label =
        ov::pass::pattern::wrap_type<ov::op::v4::Proposal>({pattern::any_input(), pattern::any_input(), reshape_label});

    matcher_pass_callback callback = [parameter_label, proposal_label](pattern::Matcher& m) -> bool {
        return crop_scales_for_proposal(m.get_pattern_value_map(), parameter_label, proposal_label);
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(proposal_label /*, matcher_name */);
    register_matcher(m, callback);
}
