// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_mvn1_to_mvn6.hpp"

#include <numeric>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertMVN1ToMVN6::ConvertMVN1ToMVN6() {
    MATCHER_SCOPE(ConvertMVN1ToMVN6);
    auto mvn = pattern::wrap_type<ov::op::v0::MVN>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto mvn_node = ov::as_type_ptr<ov::op::v0::MVN>(m.get_match_root());
        if (!mvn_node) {
            return false;
        }

        const auto input = mvn_node->input_value(0);
        auto input_rank = input.get_partial_shape().rank();
        if (!input_rank.is_static()) {
            return false;
        }
        int64_t start_axis = 1 + static_cast<int64_t>(!mvn_node->get_across_channels());
        if (input_rank.get_length() <= start_axis) {
            return false;
        }

        const auto eps_f = op::util::cast_eps_to_float(mvn_node->get_eps());

        std::vector<int64_t> axes_v(input_rank.get_length() - start_axis);
        std::iota(axes_v.begin(), axes_v.end(), start_axis);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, {axes_v.size()}, axes_v);
        auto mvn6_node = std::make_shared<ov::op::v6::MVN>(input,
                                                           axes,
                                                           mvn_node->get_normalize_variance(),
                                                           eps_f,
                                                           op::MVNEpsMode::OUTSIDE_SQRT);

        mvn6_node->set_friendly_name(mvn_node->get_friendly_name());
        ov::copy_runtime_info(mvn_node, mvn6_node);
        ov::replace_node(mvn_node, mvn6_node);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mvn, matcher_name);
    register_matcher(m, callback);
}
