// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_0d.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertGather0D::ConvertGather0D() {
    MATCHER_SCOPE(ConvertGather0D);
    auto gather = ov::pass::pattern::wrap_type<ov::op::v1::Gather>();

    matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto gather = ov::as_type_ptr<ov::op::v1::Gather>(m.get_match_root());
        if (!gather) {
            return false;
        }

        auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(gather->input_value(2).get_node_shared_ptr());
        if (!axes_constant) {
            return false;
        }

        // if the input with indices is scalar we need to unsqueeze it to 1D so plugins which do not support 0D can
        // execute this layer. Then we need to squeeze the axis dimension to restore original shape of gather output
        auto indices = gather->input_value(1);
        const auto indices_rank = indices.get_partial_shape().rank();
        if (indices_rank.is_dynamic() || indices_rank.get_length() != 0) {
            return false;
        }

        auto axis = axes_constant->cast_vector<int64_t>()[0];
        indices =
            std::make_shared<ov::op::v0::Unsqueeze>(indices, ov::op::v0::Constant::create(element::i64, Shape{1}, {0}));
        auto gather_new = std::make_shared<ov::op::v1::Gather>(gather->input_value(0), indices, axes_constant);
        auto sq = std::make_shared<ov::op::v0::Squeeze>(gather_new,
                                                        ov::op::v0::Constant::create(element::i64, Shape{1}, {axis}));
        sq->set_friendly_name(gather->get_friendly_name());

        ov::copy_runtime_info(gather, {indices.get_node_shared_ptr(), gather_new, sq});
        ov::replace_node(gather, sq);

        return true;
    };

    auto m1 = std::make_shared<ov::pass::pattern::Matcher>(gather, matcher_name);
    this->register_matcher(m1, callback);
}
