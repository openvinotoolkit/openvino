// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph_ops/tile_ie.hpp>

#include "ngraph/op/experimental/tile.hpp"

namespace ngraph {
namespace pass {

class ConvertTileToIETile;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertTileToIETile: public ngraph::pass::GraphRewrite {
public:
    ConvertTileToIETile() : GraphRewrite() {
        convert_tile();
    }

private:
    void convert_tile() {
        auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto shp = std::make_shared<pattern::op::Label>(element::i64, Shape{4});
        auto tile = std::make_shared<ngraph::op::Tile>(data, shp);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto tile = std::dynamic_pointer_cast<ngraph::op::Tile> (m.get_match_root());
            if (!tile) {
                return false;
            }

            auto data_node = tile->get_argument(0);
            auto tiles_node = std::dynamic_pointer_cast<ngraph::op::Constant> (tile->get_argument(1));
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
            auto friendly_name = tile->get_friendly_name();

            auto tiles_it = tiles.rbegin();
            while (tiles_it != tiles.rend()) {
                int64_t tile_dim = *tiles_it;
                if (tile_dim != 1) {
                    auto ie_tile = std::make_shared<ngraph::op::TileIE>(last_node, cur_dim_id, tile_dim);
                    ie_tile->set_friendly_name(friendly_name);
                    friendly_name += "_" + std::to_string(cur_dim_id);

                    last_node = std::dynamic_pointer_cast<ngraph::Node>(ie_tile);
                }
                --cur_dim_id;
                ++tiles_it;
            }

            ngraph::replace_node(m.get_match_root(), last_node);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(tile, "CPUFusion.ConvertTileToIETiles");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
