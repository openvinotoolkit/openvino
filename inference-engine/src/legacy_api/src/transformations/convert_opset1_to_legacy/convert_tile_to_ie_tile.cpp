// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/transformations/convert_opset1_to_legacy/convert_tile_to_ie_tile.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <legacy/ngraph_ops/tile_ie.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertTileToLegacyMatcher, "ConvertTileToLegacyMatcher", 0);

ngraph::pass::ConvertTileToLegacyMatcher::ConvertTileToLegacyMatcher() {
    auto tile = pattern::wrap_type<ngraph::opset1::Tile>({pattern::any_input(pattern::has_static_rank()),
                                                          pattern::wrap_type<opset1::Constant>()});

    ngraph::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto tile = std::dynamic_pointer_cast<ngraph::opset1::Tile> (m.get_match_root());
        if (!tile) {
            return false;
        }

        auto tiles_node = std::dynamic_pointer_cast<ngraph::opset1::Constant> (tile->input_value(1).get_node_shared_ptr());
        if (!tiles_node) return false;

        auto tiles = tiles_node->cast_vector<int64_t>();
        auto input_shape_rank = tile->get_input_partial_shape(0).rank().get_length();
        int64_t cur_dim_id = tiles.size() - 1;

        if (tiles.size() != input_shape_rank) return false;

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
        auto last_node = tile->input_value(0);
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
                new_ops.push_back(ie_tile);

                last_node = ie_tile;
            }
            --cur_dim_id;
            ++tiles_it;
        }

        last_node.get_node_shared_ptr()->set_friendly_name(tile->get_friendly_name());
        ngraph::copy_runtime_info(tile, new_ops);
        ngraph::replace_node(tile, {last_node});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tile, "ConvertTileToIETiles");
    this->register_matcher(m, callback);
}
