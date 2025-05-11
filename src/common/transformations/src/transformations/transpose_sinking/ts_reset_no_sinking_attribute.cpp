// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_reset_no_sinking_attribute.hpp"

#include "itt.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;

TSResetNoSinkingAttribute::TSResetNoSinkingAttribute() {
    MATCHER_SCOPE(TSResetNoSinkingAttribute);

    auto transpose_label = wrap_type<ov::op::v1::Transpose>([](const Output<Node>& output) -> bool {
        const auto& rt_info = output.get_node()->get_rt_info();
        return rt_info.find(NoTransposeSinkingAttr::get_type_info_static()) != rt_info.end();
    });
    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        const auto& transpose = pattern_to_output.at(transpose_label);
        ov::reset_no_sinking_attribute(transpose);
        return false;
    };
    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
