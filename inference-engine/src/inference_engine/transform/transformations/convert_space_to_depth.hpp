// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph/op/fused/space_to_depth.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/experimental/transpose.hpp"

namespace ngraph {
namespace pass {

class ConvertSpaceToDepth;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertSpaceToDepth: public ngraph::pass::GraphRewrite {
public:
    ConvertSpaceToDepth() : GraphRewrite() {
        convert();
    }

private:
    void convert() {
        auto input0 = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 1, 1, 1});
        auto dts = std::make_shared<ngraph::op::SpaceToDepth>(input0, ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST);

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto std_node = std::dynamic_pointer_cast<ngraph::op::SpaceToDepth> (m.get_match_root());
            if (!std_node) {
                return false;
            }

            auto input = std_node->input(0).get_source_output().get_node_shared_ptr();

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
                case ngraph::op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
                    order.push_back(1);
                    break;
                case ngraph::op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
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

            auto create_constant = [](std::vector<int64_t > & v) -> std::shared_ptr<op::Constant> {
                return op::Constant::create(element::i64, Shape{v.size()}, v);
            };

            auto reshape_begin = std::make_shared<ngraph::op::v1::Reshape>(input, create_constant(shape_begin), true);
            auto transpose = std::make_shared<ngraph::op::Transpose>(reshape_begin, create_constant(order));
            auto reshape_end = std::make_shared<ngraph::op::v1::Reshape>(transpose, create_constant(shape_end), true);
            reshape_end->set_friendly_name(std_node->get_friendly_name());
            ngraph::replace_node(std_node, reshape_end);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(dts, "ConvertSpaceToDepth");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
