// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "convert_broadcast_to_tiles.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/pattern/matcher.hpp"

#include "ngraph/graph_util.hpp"

void ngraph::pass::ConvertBroadcastToTiles::convert_broadcast_to_tiles() {
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto axs = std::make_shared<pattern::op::Label>(element::i64, Shape{1});
    auto broadcast = std::make_shared<ngraph::op::DynBroadcast>(weights, shp, axs);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto broadcast = std::dynamic_pointer_cast<ngraph::op::DynBroadcast> (m.get_match_root());

        if (!broadcast) {
            return false;
        }

        auto data_node = broadcast->get_argument(0);
        auto shape_node = std::dynamic_pointer_cast<ngraph::op::Constant> (broadcast->get_argument(1));
        auto axes_node = std::dynamic_pointer_cast<ngraph::op::Constant> (broadcast->get_argument(2));
        if (!data_node || !shape_node || !axes_node) return false;

        auto axes = axes_node->get_vector<int64_t>();
        auto output_shape = shape_node->get_vector<int64_t>();
        auto input_shape = data_node->get_shape();
        int64_t cur_dim_id = output_shape.size() - 1;
        size_t dims_count = output_shape.size();

        auto last_node = std::dynamic_pointer_cast<ngraph::Node>(data_node);

        // TODO: check that axis suits to input/output shapes

        // In case if input_hape and output_shape differ we insert Reshape to align shapes
        if (input_shape.size() != dims_count) {
            if (input_shape.size() > dims_count) {
                return false;
            }
            Shape shape(input_shape);
            for (size_t i = 0; i < (dims_count - input_shape.size()); ++i) {
                shape.insert(shape.begin(), 1);
            }
            auto shape_const = std::make_shared<ngraph::op::Constant>(element::i64, Shape{shape.size()}, shape);
            auto reshape = std::make_shared<ngraph::op::DynReshape>(data_node, shape_const);  // TODO: use Reshape instead
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

        auto const_node = std::make_shared<ngraph::op::Constant>(element::i64, Shape{dims_count}, dims);
        auto tile = std::make_shared<ngraph::op::Tile>(last_node, const_node);
        tile->set_friendly_name(broadcast->get_friendly_name());

        last_node = std::dynamic_pointer_cast<ngraph::Node>(tile);

        ngraph::replace_node(m.get_match_root(), last_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(broadcast, "CPUFusion.ConvertBroadcastToTile");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
