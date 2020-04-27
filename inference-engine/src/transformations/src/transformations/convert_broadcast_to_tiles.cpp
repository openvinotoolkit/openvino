// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_broadcast_to_tiles.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertBroadcastToTiles::convert_broadcast_to_tiles() {
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape {1});
    auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape {1});
    auto axs = std::make_shared<pattern::op::Label>(element::i64, Shape {1});
    auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(weights, shp, axs);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto broadcast = std::dynamic_pointer_cast<ngraph::opset1::Broadcast>(m.get_match_root());

        if (!broadcast) {
            return false;
        }

        auto data_node = broadcast->get_argument(0);
        auto shape_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(broadcast->get_argument(1));
        auto axes_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(broadcast->get_argument(2));
        if (!data_node || !shape_node || !axes_node) return false;

        auto output_shape = shape_node->get_vector<int64_t>();
        auto input_shape = data_node->get_shape();
        int64_t cur_dim_id = output_shape.size() - 1;
        size_t dims_count = output_shape.size();

        auto last_node = std::dynamic_pointer_cast<ngraph::Node>(data_node);

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
                auto axes = axes_node->get_vector<int64_t>();
                shape.assign(output_shape.size(), 1);
                for (size_t i = 0; i < input_shape.size(); ++i) {
                    shape[axes[i]] = input_shape[i];
                }
            } else {
                return false;
            }
            auto shape_const = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape {shape.size()}, shape);
            auto reshape = std::make_shared<ngraph::opset1::Reshape>(data_node, shape_const, true);
            new_ops.push_back(reshape);
            last_node = std::dynamic_pointer_cast<ngraph::Node>(reshape);
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

        auto const_node = std::make_shared<ngraph::opset1::Constant>(element::i64, Shape {dims_count}, dims);
        auto tile = std::make_shared<ngraph::opset1::Tile>(last_node, const_node);
        new_ops.push_back(tile);
        tile->set_friendly_name(broadcast->get_friendly_name());

        last_node = std::dynamic_pointer_cast<ngraph::Node>(tile);
        ngraph::copy_runtime_info(broadcast, new_ops);
        ngraph::replace_node(broadcast, last_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(broadcast, "ConvertBroadcastToTile");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
