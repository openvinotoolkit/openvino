// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"

#include <openvino/core/validation_util.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/gather_elements.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <ov_ops/rotary_positional_embeddings.hpp>

#include "itt.hpp"

ov::pass::RPE_Optimization::RPE_Optimization() {
    MATCHER_SCOPE(RPE_Optimization);

    auto sin = pattern::wrap_type<op::v6::GatherElements, op::v0::Unsqueeze>();  // any_input doesn't work here
    auto cos = pattern::wrap_type<op::v6::GatherElements, op::v0::Unsqueeze>();  // any_input doesn't work here

    auto source = pattern::any_input(pattern::has_static_rank());

    // rotate_half begin
    auto split_length =
        pattern::wrap_type<op::v0::Constant>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
            const auto& constant = ov::as_type_ptr<op::v0::Constant>(n);
            if (!constant)
                return false;
            const auto& value = constant->cast_vector<int64_t>();
            return value.size() == 2 && value[0] == value[1];
        }));  // make sure constant contains 2 elements with same content; it may be -1, but we fix it earlier
    auto axis = pattern::any_input();
    auto vsplit = pattern::wrap_type<op::v1::VariadicSplit>({source, axis, split_length});
    vsplit->set_output_size(2);
    auto minus_1 =
        pattern::wrap_type<op::v0::Constant>(pattern::op::as_value_predicate([](std::shared_ptr<Node> n) -> bool {
            const auto& constant = ov::as_type_ptr<op::v0::Constant>(n);
            if (!constant)
                return false;
            const auto& value = constant->cast_vector<int64_t>();
            return value.size() == 1 && value[0] == -1;
        }));  // make sure it is == -1
    auto neg = pattern::wrap_type<op::v1::Multiply>({vsplit->output(1), minus_1});
    auto concat = pattern::wrap_type<op::v0::Concat>({neg, vsplit->output(0)});  // make sure axis eq to vsplit eq -1
    // rotate half end
    auto mul_sin = pattern::wrap_type<op::v1::Multiply>({concat, sin});
    auto mul_cos = pattern::wrap_type<op::v1::Multiply>({source, cos});
    auto add = pattern::wrap_type<op::v1::Add>({mul_cos, mul_sin});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        auto value_map = m.get_pattern_value_map();

        auto input = value_map.at(source);
        auto concat_node = ov::as_type_ptr<op::v0::Concat>(value_map.at(concat).get_node_shared_ptr());
        if (!concat_node)
            return false;
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto split_axis_node = ov::get_constant_from_source(value_map.at(axis));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!split_axis_node)
            return false;
        auto value = split_axis_node->cast_vector<int64_t>();
        if (value.size() != 1)
            return false;
        auto concat_axis = concat_node->get_concatenation_axis();
        concat_axis = concat_axis < 0 ? concat_axis + input.get_partial_shape().size() : concat_axis;
        auto split_axis = value[0];
        split_axis = split_axis < 0 ? split_axis + input.get_partial_shape().size() : split_axis;
        if (concat_axis != split_axis)
            return false;
        auto rope = std::make_shared<ov::op::internal::RPE>(input,
                                                            value_map.at(sin),
                                                            value_map.at(cos),
                                                            concat_node->get_axis());
        value_map.at(add).replace(rope->output(0));  // TODO: update fused names
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(add, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
