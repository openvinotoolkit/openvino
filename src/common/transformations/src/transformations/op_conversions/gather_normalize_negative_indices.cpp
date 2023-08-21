// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/gather_normalize_negative_indices.hpp"

#include <memory>
#include <openvino/core/rt_info.hpp>
#include <openvino/core/validation_util.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"

ov::pass::GatherNegativeConstIndicesNormalize::GatherNegativeConstIndicesNormalize() {
    MATCHER_SCOPE(GatherNegativeConstIndicesNormalize);
    auto data_input = pattern::any_input(pattern::has_static_rank());
    auto axis_input = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto indices_input = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto gather_node = ov::pass::pattern::wrap_type<op::util::GatherBase>({data_input, indices_input, axis_input});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto gather = pattern_to_output.at(gather_node).get_node_shared_ptr();
        auto data = pattern_to_output.at(data_input);
        auto axis_constant =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output.at(axis_input).get_node_shared_ptr());
        auto indices_constant =
            std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_to_output.at(indices_input).get_node_shared_ptr());

        if (!gather || !axis_constant || !indices_constant) {
            return false;
        }

        auto indices = indices_constant->cast_vector<int64_t>();
        if (indices.size() != 1 || indices[0] >= 0) {
            return false;
        }

        auto axis = axis_constant->cast_vector<int64_t>();
        if (axis.size() != 1) {
            return false;
        }

        auto axis_value = axis[0];

        // normalize `axis` value if it is negative
        if (axis_value < 0) {
            axis_value = axis_value + data.get_partial_shape().rank().get_length();
        }

        if (data.get_partial_shape().rank().get_length() < axis_value) {
            return false;
        }

        // check `axis` dimension of data tensor is static
        if (!data.get_partial_shape()[axis_value].is_static()) {
            return false;
        }

        auto input_type = indices_constant->get_element_type();
        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data, input_type);
        auto input_gather =
            std::make_shared<ov::op::v7::Gather>(shape_of,
                                                 ov::op::v0::Constant::create(input_type, Shape{}, {axis_value}),
                                                 ov::op::v0::Constant::create(input_type, Shape{}, {0}));

        std::shared_ptr<Node> add = std::make_shared<ov::op::v1::Add>(input_gather, indices_constant);
        OPENVINO_SUPPRESS_DEPRECATED_START
        if (auto folded_const = ov::get_constant_from_source(add)) {
            OPENVINO_SUPPRESS_DEPRECATED_END
            add = folded_const;
        }
        gather->input(1).replace_source_output(add);

        ov::copy_runtime_info(gather, {shape_of, input_gather, add});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gather_node, matcher_name);
    register_matcher(m, callback);
}
