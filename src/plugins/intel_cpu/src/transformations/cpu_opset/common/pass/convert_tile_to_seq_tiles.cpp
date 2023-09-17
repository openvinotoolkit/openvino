// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_tile_to_seq_tiles.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <ngraph/rt_info.hpp>

#include "itt.hpp"

ov::intel_cpu::ConvertTileToSeqTiles::ConvertTileToSeqTiles() {
    MATCHER_SCOPE(ConvertTileToSeqTiles);
    auto tile = ngraph::pattern::wrap_type<ngraph::opset1::Tile>({ngraph::pattern::any_input(ngraph::pattern::has_static_rank()),
                                                                  ngraph::pattern::wrap_type<ngraph::opset1::Constant>()});

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto tile = std::dynamic_pointer_cast<ngraph::opset1::Tile>(m.get_match_root());
        if (!tile) {
            return false;
        }

        auto tiles_node = std::dynamic_pointer_cast<ngraph::opset1::Constant>(tile->input_value(1).get_node_shared_ptr());
        if (!tiles_node) return false;

        auto tiles = tiles_node->cast_vector<int64_t>();
        auto input_shape_rank = static_cast<size_t>(tile->get_input_partial_shape(0).rank().get_length());
        int64_t cur_dim_id = tiles.size() - 1;

        if (tiles.size() != input_shape_rank) return false;

        auto last_node = tile->input_value(0);
        auto friendly_name = tile->get_friendly_name();

        int num_of_tile_dims = 0;
        for (auto t : tiles) {
            if (t != 1) {
                num_of_tile_dims++;
            }
        }

        if (num_of_tile_dims == 0) {
            auto outputs = tile->get_output_target_inputs(0);
            for (const auto &out : outputs) {
                if (std::dynamic_pointer_cast<ngraph::opset1::Result>(out.get_node()->shared_from_this())) {
                    return false;
                }
            }
            ngraph::replace_node(tile, {last_node});
            return true;
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

        ngraph::NodeVector new_ops;

        auto tiles_it = tiles.rbegin();
        while (tiles_it != tiles.rend()) {
            int64_t tile_dim = *tiles_it;
            if (tile_dim != 1) {
                std::vector<int64_t> dims(input_shape_rank, 1);
                dims[cur_dim_id] = tile_dim;
                auto const_node = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{input_shape_rank}, dims);
                auto new_tile = std::make_shared<ngraph::opset1::Tile>(last_node, const_node);
                new_tile->set_friendly_name(friendly_name);
                friendly_name += "_" + std::to_string(cur_dim_id);
                new_ops.push_back(new_tile);

                last_node = new_tile;
            }
            --cur_dim_id;
            ++tiles_it;
        }

        last_node.get_node_shared_ptr()->set_friendly_name(tile->get_friendly_name());
        ngraph::copy_runtime_info(tile, new_ops);
        ngraph::replace_node(tile, {last_node});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(tile, matcher_name);
    this->register_matcher(m, callback);
}
