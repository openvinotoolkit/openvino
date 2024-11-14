// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_broadcast_to_tiles.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ConvertBroadcastToTiles::ConvertBroadcastToTiles() {
    MATCHER_SCOPE(ConvertBroadcastToTiles);
    auto broadcast = ov::pass::pattern::wrap_type<ov::op::v1::Broadcast>();

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto broadcast = ov::as_type_ptr<ov::op::v1::Broadcast>(m.get_match_root());

        if (!broadcast) {
            return false;
        }

        auto data_node = broadcast->input_value(0);
        if (data_node.get_partial_shape().is_dynamic()) {
            return false;
        }

        auto shape_node = ov::as_type_ptr<ov::op::v0::Constant>(broadcast->input_value(1).get_node_shared_ptr());
        auto axes_node = ov::as_type_ptr<ov::op::v0::Constant>(broadcast->input_value(2).get_node_shared_ptr());
        if (!shape_node || !axes_node)
            return false;

        auto output_shape = shape_node->cast_vector<int64_t>();
        auto input_shape = data_node.get_shape();
        int64_t cur_dim_id = output_shape.size() - 1;
        size_t dims_count = output_shape.size();

        auto last_node = data_node;

        NodeVector new_ops;

        // In case if input_shape and output_shape differ we insert Reshape to align shapes
        if (input_shape.size() != dims_count) {
            if (input_shape.size() > dims_count) {
                return false;
            }
            Shape shape;
            auto broadcast_type = broadcast->get_broadcast_spec();
            if (broadcast_type == op::AutoBroadcastType::NUMPY) {
                shape = input_shape;
                for (size_t i = 0; i < (dims_count - input_shape.size()); ++i) {
                    shape.insert(shape.begin(), 1);
                }
            } else if (broadcast_type == op::AutoBroadcastType::NONE) {
                auto axes = axes_node->cast_vector<int64_t>();
                shape.assign(output_shape.size(), 1);
                for (size_t i = 0; i < input_shape.size(); ++i) {
                    shape[axes[i]] = input_shape[i];
                }
            } else {
                return false;
            }
            auto shape_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{shape.size()}, shape);
            auto reshape = std::make_shared<ov::op::v1::Reshape>(data_node, shape_const, true);
            new_ops.push_back(reshape);
            last_node = reshape;
            input_shape = shape;
        }

        std::vector<int64_t> dims(dims_count, 1);
        auto input_shape_it = input_shape.rbegin();
        auto output_shape_it = output_shape.rbegin();
        while (output_shape_it != output_shape.rend() && input_shape_it != input_shape.rend()) {
            int64_t in_dim = *input_shape_it, out_dim = *output_shape_it;
            if (in_dim != out_dim) {
                if (in_dim != 1) {
                    return false;
                }
                dims[cur_dim_id] = out_dim;
            }

            --cur_dim_id;
            ++output_shape_it;
            ++input_shape_it;
        }

        auto const_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{dims_count}, dims);
        auto tile = register_new_node<ov::op::v0::Tile>(last_node, const_node);
        new_ops.push_back(tile);
        tile->set_friendly_name(broadcast->get_friendly_name());

        ov::copy_runtime_info(broadcast, new_ops);
        ov::replace_node(broadcast, tile);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(broadcast, matcher_name);
    this->register_matcher(m, callback);
}
