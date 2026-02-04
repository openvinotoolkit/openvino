// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_space_to_depth.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
ov::pass::ConvertSpaceToDepth::ConvertSpaceToDepth() {
    MATCHER_SCOPE(ConvertSpaceToDepth);
    auto dts = ov::pass::pattern::wrap_type<v0::SpaceToDepth>(
        {ov::pass::pattern::any_input(ov::pass::pattern::has_static_shape())});

    matcher_pass_callback callback = [this](Matcher& m) {
        auto std_node = ov::as_type_ptr<v0::SpaceToDepth>(m.get_match_root());
        if (!std_node || transformation_callback(std_node)) {
            return false;
        }

        auto input = std_node->input_value(0);

        /*
         * In this transformation we decompose SpaceToDepth operation to the next sequence of ops:
         * Reshape(shape_begin)->Transpose(order)->Reshape(shape_end)
         *
         * if mode equal to blocks_first
         * transpose_order = [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K]
         *
         * if mode equal to depth_first
         * transpose_order = [0,  1, 3, 5, ..., K + (K + 1),  2, 4, ..., K + K]
         *
         */

        auto input_shape = std_node->input(0).get_shape();
        auto spatial_dims = input_shape.size() - 2;
        auto block_size = std_node->get_block_size();
        auto mode = std_node->get_mode();

        // Calculate Reshape shape_begin
        std::vector<int64_t> shape_begin{static_cast<int64_t>(input_shape[0]), static_cast<int64_t>(input_shape[1])};
        for (size_t i = 0; i < spatial_dims; ++i) {
            shape_begin.push_back(input_shape[2 + i] / block_size);
            shape_begin.push_back(block_size);
        }

        // Calculate Transpose order
        std::vector<int64_t> order{0};
        for (size_t i = 0, j = 3; i < spatial_dims; ++i, j += 2) {
            order.push_back(j);
        }

        switch (mode) {
        case v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
            order.push_back(1);
            break;
        case v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
            order.insert(order.begin() + 1, 1);
            break;
        }

        for (size_t i = 0, j = 2; i < spatial_dims; ++i, j += 2) {
            order.push_back(j);
        }

        // Calculate Reshape shape_end
        std::vector<int64_t> shape_end{static_cast<int64_t>(input_shape[0])};
        auto C = input_shape[1];
        for (size_t i = 0; i < spatial_dims; ++i) {
            shape_end.push_back(input_shape[2 + i] / block_size);
            C *= block_size;
        }
        shape_end.insert(shape_end.begin() + 1, C);

        auto create_constant = [](std::vector<int64_t>& v) -> std::shared_ptr<v0::Constant> {
            return v0::Constant::create(element::i64, Shape{v.size()}, v);
        };

        auto reshape_begin = std::make_shared<v1::Reshape>(input, create_constant(shape_begin), true);
        auto transpose = std::make_shared<v1::Transpose>(reshape_begin, create_constant(order));
        auto reshape_end = std::make_shared<v1::Reshape>(transpose, create_constant(shape_end), true);
        reshape_end->set_friendly_name(std_node->get_friendly_name());
        ov::copy_runtime_info(std_node, {reshape_begin, transpose, reshape_end});
        ov::replace_node(std_node, reshape_end);
        return true;
    };

    auto m = std::make_shared<Matcher>(dts, matcher_name);
    this->register_matcher(m, callback);
}
