// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/disable_fp16_comp_ltx_rope.hpp"

#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace op_util = ov::op::util;

namespace {

// Frequencies may be computed in the graph (e.g. theta^linspace in f64, sized by a ShapeOf): accept any subgraph
// fed by Constants and shapes only.
bool is_constant_subgraph(const ov::Output<ov::Node>& output) {
    std::unordered_set<ov::Node*> visited;
    std::vector<ov::Node*> stack{output.get_node()};
    while (!stack.empty()) {
        auto node = stack.back();
        stack.pop_back();
        if (!visited.insert(node).second || ov::is_type_any_of<v0::Constant, op_util::ShapeOfBase>(node)) {
            continue;
        }
        if (node->get_input_size() == 0) {
            return false;
        }
        for (const auto& input : node->input_values()) {
            stack.push_back(input.get_node());
        }
    }
    return true;
}

// Marks the matched table chain and the input paths of the grid and the frequencies; skips the match if they depend
// on activations.
bool mark_rope_table(const ov::NodeVector& chain, const std::vector<ov::Node*>& inputs) {
    bool depends_on_activations = false;
    std::vector<ov::Node*> path;
    std::unordered_set<ov::Node*> visited;
    for (const auto& input : inputs) {
        op_util::visit_path(
            input,
            visited,
            [&path](ov::Node* node) {
                path.push_back(node);
            },
            [&depends_on_activations](ov::Node* node) {
                if (ov::is_type_any_of<v0::MatMul, op_util::ConvolutionBase>(node)) {
                    depends_on_activations = true;
                    return true;
                }
                return ov::is_type_any_of<v0::Constant, v0::Parameter, op_util::ShapeOfBase>(node);
            });
    }
    if (depends_on_activations) {
        return false;
    }
    for (const auto& node : chain) {
        ov::disable_conversion(node, ov::element::f16);
    }
    for (const auto& node : path) {
        ov::disable_conversion(node->shared_from_this(), ov::element::f16);
    }
    return true;
}

}  // namespace

ov::pass::DisableFP16CompForLtxVideoRopePattern::DisableFP16CompForLtxVideoRopePattern() {
    MATCHER_SCOPE(DisableFP16CompForLtxVideoRopePattern);
    using namespace ov::pass::pattern;

    auto grid = any_input();
    auto freqs = any_input(is_constant_subgraph);
    auto mul = wrap_type<v1::Multiply>({grid, freqs});
    auto shift = wrap_type<v1::Add>({mul, wrap_const()});
    auto layout = optional<v1::Transpose, v1::Reshape>({shift, any_input()});
    auto flatten = optional<v1::Reshape>({layout, any_input()});
    auto sin_or_cos = wrap_type<v0::Sin, v0::Cos>({flatten});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        ov::NodeVector chain;
        for (const auto& label : {mul, shift, layout, flatten, sin_or_cos}) {
            if (pattern_map.count(label)) {
                chain.push_back(pattern_map.at(label).get_node_shared_ptr());
            }
        }
        mark_rope_table(chain, {pattern_map.at(grid).get_node(), pattern_map.at(freqs).get_node()});
        // markup only: the graph itself is not changed
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos, matcher_name);
    register_matcher(m, callback);
}

ov::pass::DisableFP16CompForLtx2RopePattern::DisableFP16CompForLtx2RopePattern() {
    MATCHER_SCOPE(DisableFP16CompForLtx2RopePattern);
    using namespace ov::pass::pattern;

    auto grid = any_input();
    auto shift = wrap_type<v1::Add>({grid, wrap_const()});
    // MoveEltwiseUpThroughDataMov may move the shift above the position axis selection
    auto squeeze = optional<v0::Squeeze>({shift, wrap_const()});
    auto select = optional<v8::Gather>({squeeze, wrap_const(), wrap_const()});
    auto unsqueeze = optional<v0::Unsqueeze>({select, wrap_const()});
    auto freqs = any_input(is_constant_subgraph);
    auto mul = wrap_type<v1::Multiply>({unsqueeze, freqs});
    auto layout = optional<v1::Transpose, v1::Reshape>({mul, any_input()});
    auto flatten = optional<v1::Reshape>({layout, any_input()});
    auto sin_or_cos = wrap_type<v0::Sin, v0::Cos>({flatten});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        ov::NodeVector chain;
        for (const auto& label : {shift, squeeze, select, unsqueeze, mul, layout, flatten, sin_or_cos}) {
            if (pattern_map.count(label)) {
                chain.push_back(pattern_map.at(label).get_node_shared_ptr());
            }
        }
        mark_rope_table(chain, {pattern_map.at(grid).get_node(), pattern_map.at(freqs).get_node()});
        // markup only: the graph itself is not changed
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos, matcher_name);
    register_matcher(m, callback);
}

ov::pass::DisableFP16CompForLtxRope::DisableFP16CompForLtxRope() {
    ADD_MATCHER_FOR_THIS(DisableFP16CompForLtxVideoRopePattern)
    ADD_MATCHER_FOR_THIS(DisableFP16CompForLtx2RopePattern)
}
