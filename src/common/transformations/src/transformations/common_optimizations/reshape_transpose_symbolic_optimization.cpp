// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reshape_transpose_symbolic_optimization.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/symbolic_transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ReshapeTransposeSymbolicOptimization::ReshapeTransposeSymbolicOptimization() {
    MATCHER_SCOPE(ReshapeTransposeSymbolicOptimization);
    using namespace ov::op;
    using namespace ov::pass::pattern;

    auto in = ov::pass::pattern::wrap_type<v1::Reshape, v1::Transpose>();
    auto pattern_root = ov::pass::pattern::wrap_type<v1::Reshape, v1::Transpose>({in, any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        using namespace ov::symbol::util;

        std::shared_ptr<ov::Node> root_node = m.get_match_root();
        if (!root_node) {
            return false;
        }

        auto is_reshape_transpose = [](const std::shared_ptr<ov::Node>& node) -> bool {
            return ov::is_type<v1::Reshape>(node) || ov::is_type<v1::Transpose>(node);
        };

        std::shared_ptr<ov::Node> cur_node;
        std::shared_ptr<ov::Node> next_node = root_node->input_value(0).get_node_shared_ptr();
        while (is_reshape_transpose(next_node)) {
            cur_node = next_node;
            next_node = next_node->input_value(0).get_node_shared_ptr();
        }

        if (!cur_node) {
            return false;
        }

        auto cur_ps = cur_node->input_value(0).get_partial_shape();
        std::vector<ov::Dimension> root_ps;
        for (const auto& dim : root_node->output(0).get_partial_shape()) {
            root_ps.push_back(dim);
        }

        std::vector<int64_t> pos;
        int64_t order = 0;
        bool in_order = true;
        for (const auto& cur_dim : cur_ps) {
            auto corresponding_sym = std::find_if(root_ps.begin(), root_ps.end(), [&](const ov::Dimension& root_dim) {
                return dims_are_equal(cur_dim, root_dim);
            });

            if (corresponding_sym == root_ps.end()) {
                return false;
            }

            auto cur_pos = std::distance(root_ps.begin(), corresponding_sym);
            in_order = (cur_pos == order++) && in_order;
            pos.push_back(cur_pos);

            root_ps.erase(corresponding_sym);
        }

        if (!root_ps.empty()) {
            return false;
        }

        if (in_order) {
            // the subgraph does nothing
            ov::replace_output_update_name(root_node->output(0), cur_node->input_value(0));
        } else {
            auto transpose_order = ov::op::v0::Constant::create(ov::element::i64, Shape{pos.size()}, pos);
            auto transpose = std::make_shared<ov::op::v1::Transpose>(cur_node->input_value(0), transpose_order);
            ov::replace_node(root_node, transpose);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_root, matcher_name);
    this->register_matcher(m, callback);
}
