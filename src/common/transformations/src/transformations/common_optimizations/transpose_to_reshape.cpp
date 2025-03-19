// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/transpose_to_reshape.hpp"

#include <memory>
#include <numeric>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

ov::pass::TransposeToReshape::TransposeToReshape() {
    MATCHER_SCOPE(TransposeToReshape);

    auto transpose_label = pattern::wrap_type<ov::op::v1::Transpose>(
        {pattern::any_input(pattern::has_static_rank()), pattern::wrap_type<ov::op::v0::Constant>()});
    ov::matcher_pass_callback matcher_pass_callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto transpose = m.get_match_root();
        auto data = transpose->input_value(0);
        const auto input_shape = transpose->input(0).get_partial_shape();

        const size_t input_shape_rank = input_shape.rank().get_length();

        auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        if (!order || !ov::shape_size(order->get_shape())) {
            return false;
        }

        const auto order_value = order->cast_vector<int64_t>();

        // Check that transpose order without 1 dims has an ascending order
        int64_t last_dim(-1);
        for (size_t i = 0; i < input_shape_rank; ++i) {
            if (input_shape[order_value[i]].is_dynamic() || input_shape[order_value[i]] != 1) {
                if (order_value[i] < last_dim) {
                    return false;
                }
                last_dim = order_value[i];
            }
        }

        // Transpose operation can be removed if original transpose order is sorted
        // or dimension that changes their places equal to 1
        using DimensionToPosition = struct {
            Dimension dim;
            size_t pos;
        };
        std::vector<DimensionToPosition> dims;
        for (size_t i = 0; i < input_shape_rank; ++i) {
            if (order_value[i] != static_cast<int64_t>(i)) {
                dims.push_back({input_shape[order_value[i]], i});
            }
        }

        // If number of dimensions != 1 to move equal to 0 we can remove this Transpose
        if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
                return !(item.dim.is_static() && item.dim.get_length() == 1);
            }) == 0) {
            return replace_output_update_name(transpose->output(0), transpose->input_value(0));
        }

        // Transpose can be replaced with Reshape in two ways:
        // 1. Reshape with dims as Constant
        // 2. Reshape with dims as input (ShapeOf->Gather)
        //
        // The first case is possible only if one or less dynamic dimensions changes their position
        // For example: input_shape {?, 3, 1, ?} and order {0, 1, 3, 2} can be replaced with Reshape
        // with Constant {0, 3, -1, 1} but if input_shape {?, 1, 1, ?} and order {1, 0, 3, 2} transpose
        // cannot be replaced int the same way and in this case its only possible to use Gather(ShapeOf,
        // order)

        Output<Node> reshape_dim;
        NodeVector new_ops;

        if (count_if(dims.begin(), dims.end(), [](const DimensionToPosition& item) {
                return item.dim.is_dynamic();
            }) < 2) {
            std::vector<int64_t> reshape_value(input_shape_rank, 0);
            for (const auto& item : dims) {
                reshape_value[item.pos] = item.dim.is_dynamic() ? -1 : item.dim.get_length();
            }
            reshape_dim = ov::op::v0::Constant::create(element::i64, Shape{reshape_value.size()}, reshape_value);
        } else {
            auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(data);
            new_ops.push_back(shape_of);
            reshape_dim =
                std::make_shared<ov::op::v1::Gather>(shape_of,
                                                     order,
                                                     ov::op::v0::Constant::create(element::i64, Shape{1}, {0}));
            new_ops.push_back(reshape_dim.get_node_shared_ptr());
        }

        auto reshape_op = register_new_node<ov::op::v1::Reshape>(data, reshape_dim, true);
        new_ops.push_back(reshape_op);

        reshape_op->set_friendly_name(transpose->get_friendly_name());
        copy_runtime_info(transpose, new_ops);
        replace_node(transpose, reshape_op);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
