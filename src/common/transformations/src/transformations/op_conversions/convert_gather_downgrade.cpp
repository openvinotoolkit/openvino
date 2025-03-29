// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_gather_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace std;
using namespace ov;

pass::ConvertGather7ToGather1::ConvertGather7ToGather1() {
    MATCHER_SCOPE(ConvertGather7ToGather1);

    auto gather_v7_pattern = pattern::wrap_type<ov::op::v7::Gather>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v7_node = ov::as_type_ptr<ov::op::v7::Gather>(m.get_match_root());
        if (!gather_v7_node)
            return false;
        if (gather_v7_node->get_batch_dims() != 0)
            return false;

        auto gather_v1_node = make_shared<ov::op::v1::Gather>(gather_v7_node->input_value(0),
                                                              gather_v7_node->input_value(1),
                                                              gather_v7_node->input_value(2));

        gather_v1_node->set_friendly_name(gather_v7_node->get_friendly_name());
        ov::copy_runtime_info(gather_v7_node, gather_v1_node);
        ov::replace_node(gather_v7_node, gather_v1_node);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(gather_v7_pattern, matcher_name);
    register_matcher(m, callback);
}

pass::ConvertGather8ToGather7::ConvertGather8ToGather7() {
    MATCHER_SCOPE(ConvertGather8ToGather7);

    auto gather_v8_pattern = pattern::wrap_type<ov::op::v8::Gather>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto gather_v8_node = ov::as_type_ptr<ov::op::v8::Gather>(m.get_match_root());
        if (!gather_v8_node)
            return false;

        auto data = gather_v8_node->input_value(0);
        auto indices_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(gather_v8_node->input_value(1).get_node_shared_ptr());
        auto axis_constant =
            ov::as_type_ptr<ov::op::v0::Constant>(gather_v8_node->input_value(2).get_node_shared_ptr());
        if (!indices_constant || !axis_constant)
            return false;

        auto axis = axis_constant->cast_vector<int64_t>();
        if (axis.size() != 1) {
            return false;
        }
        auto axis_value = axis[0];
        // normalize `axis` value if it is negative
        if (axis_value < 0) {
            if (!data.get_partial_shape().rank().is_static()) {
                return false;
            }
            axis_value = axis_value + data.get_partial_shape().rank().get_length();
        }
        if (data.get_partial_shape().rank().get_length() < axis_value) {
            return false;
        }
        // check `axis` dimension of data tensor is static
        if (!data.get_partial_shape()[axis_value].is_static()) {
            return false;
        }
        auto axis_dim = data.get_partial_shape()[axis_value].get_length();

        auto indices = indices_constant->cast_vector<int64_t>();
        // Check all the indices are not out of bound and check whether normalization is possible for negative values
        bool do_indices_normalization = false;
        for (size_t i = 0; i < indices.size(); i++) {
            if (indices[i] < -axis_dim || indices[i] >= axis_dim) {
                return false;
            }
            if (indices[i] < 0) {
                do_indices_normalization = true;
                indices[i] = indices[i] + axis_dim;
            }
        }

        std::shared_ptr<ov::Node> new_indices_constant;
        if (do_indices_normalization) {
            new_indices_constant = std::make_shared<ov::op::v0::Constant>(indices_constant->get_element_type(),
                                                                          indices_constant->get_shape(),
                                                                          indices);
        } else {
            new_indices_constant = indices_constant;
        }

        auto gather_v7_node = make_shared<ov::op::v7::Gather>(gather_v8_node->input_value(0),
                                                              new_indices_constant,
                                                              gather_v8_node->input_value(2),
                                                              gather_v8_node->get_batch_dims());

        gather_v7_node->set_friendly_name(gather_v8_node->get_friendly_name());
        ov::copy_runtime_info(gather_v8_node, gather_v7_node);
        ov::replace_node(gather_v8_node, gather_v7_node);
        return true;
    };

    auto m = make_shared<pattern::Matcher>(gather_v8_pattern, matcher_name);
    register_matcher(m, callback);
}
