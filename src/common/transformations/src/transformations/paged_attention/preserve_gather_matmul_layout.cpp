// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/preserve_gather_matmul_layout.hpp"

#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/gather_matmul.hpp"

ov::pass::PreserveGatherMatmulLayout::PreserveGatherMatmulLayout() {
    MATCHER_SCOPE(PreserveGatherMatmulLayout);
    using namespace ov::pass::pattern;

    auto shared_activation = any_input(op::Predicate(
        [](const ov::Output<ov::Node>& output) {
            const auto& shape = output.get_partial_shape();
            if (shape.rank().is_dynamic() || shape.rank().get_length() != 3 || shape[0] != 1)
                return false;
            const auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output.get_node_shared_ptr());
            if (!reshape || !reshape->get_special_zero())
                return true;
            const auto pattern = ov::as_type_ptr<ov::op::v0::Constant>(reshape->get_input_node_shared_ptr(1));
            return !pattern || pattern->cast_vector<int64_t>() != std::vector<int64_t>{1, -1, 0};
        },
        "shared_expert_activation_without_layout_reshape"));
    auto gmm = wrap_type<ov::op::internal::GatherMatmul>({shared_activation, any_input(), any_input(), any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto node = m.get_match_root();
        if (transformation_callback(node))
            return false;
        const auto pattern = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, -1, 0});
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(node->input_value(0), pattern, true);
        ov::copy_runtime_info(node, {pattern, reshape});
        node->input(0).replace_source_output(reshape);
        return true;
    };
    register_matcher(std::make_shared<Matcher>(gmm, matcher_name), callback);
}
