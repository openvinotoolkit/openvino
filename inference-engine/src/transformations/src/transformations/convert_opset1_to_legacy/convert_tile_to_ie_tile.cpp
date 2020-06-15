// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include <ngraph_ops/tile_ie.hpp>
#include <ngraph/rt_info.hpp>

void ngraph::pass::ConvertTileToIETileMatcher::register_matcher(std::shared_ptr<ngraph::pass::GraphRewrite> t) {
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
    auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
    auto tile = std::make_shared<ngraph::opset1::Tile>(data, shp);

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto tile = std::dynamic_pointer_cast<ngraph::opset1::Tile> (m.get_match_root());
        if (!tile) {
            return false;
        }

        auto data_node = tile->get_argument(0);
        auto tiles_node = std::dynamic_pointer_cast<ngraph::opset1::Constant> (tile->get_argument(1));
        if (!data_node || !tiles_node) return false;

        auto tiles = tiles_node->get_vector<int64_t>();
        auto input_shape = data_node->get_shape();
        int64_t cur_dim_id = tiles.size() - 1;

        if (tiles.size() != input_shape.size()) return false;

        // IE Tile operations supports only one axis to be tiled
        // bool already_set = false;
        // int64_t axis, tiles;
        // for (size_t i = 0; i < input_shape.size(); ++i) {
        //     if (shape[i] != 1) {
        //         if (already_set) return false;
        //         axis = i;
        //         tiles = shape[i];
        //         already_set = true;
        //     }
        // }
        //
        // if (!already_set) return false;
        auto last_node = std::dynamic_pointer_cast<ngraph::Node>(data_node);
        if (!last_node)
            return false;
        auto friendly_name = tile->get_friendly_name();

        int num_of_tile_dims = 0;
        for (auto t : tiles) {
            if (t != 1) {
                num_of_tile_dims++;
            }
        }
        // Will generate sequence of Tile operations if num_of_tile_dims != 1
        // because IE Tile operations supports only one axis to be tiled.
        // To keep op name unique will use special IE specific delimiter ':'
        // Original frameworks doesn't use such delimiter in names, so it will
        // guarantee that newly generated name like "original_name:_1" doesn't
        // match with already existed names.
        if (num_of_tile_dims > 1) {
            friendly_name += ":";
        }

        NodeVector new_ops;

        auto tiles_it = tiles.rbegin();
        while (tiles_it != tiles.rend()) {
            int64_t tile_dim = *tiles_it;
            if (tile_dim != 1) {
                auto ie_tile = std::make_shared<ngraph::op::TileIE>(last_node, cur_dim_id, tile_dim);
                ie_tile->set_friendly_name(friendly_name);
                friendly_name += "_" + std::to_string(cur_dim_id);

                last_node = std::dynamic_pointer_cast<ngraph::Node>(ie_tile);
                new_ops.push_back(last_node);
            }
            --cur_dim_id;
            ++tiles_it;
        }

        last_node->set_friendly_name(tile->get_friendly_name());
        ngraph::copy_runtime_info(tile, new_ops);
        ngraph::replace_node(tile, last_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tile, "CPUFusion.ConvertTileToIETiles");
    t->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
