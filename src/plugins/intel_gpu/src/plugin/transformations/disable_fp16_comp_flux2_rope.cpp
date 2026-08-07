// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_flux2_rope.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::intel_gpu {

namespace {

void mark_path_from_output(ov::Node* root, std::unordered_set<ov::Node*>& visited, const std::function<bool(ov::Node*)>& skip_node_predicate) {
    if (!root || visited.count(root))
        return;
    auto visit_func = [](ov::Node* node) {
        ov::disable_conversion(node->shared_from_this(), ov::element::f16);
    };
    op_util::visit_path(root, visited, visit_func, skip_node_predicate);
}

auto skip_node_predicate() {
    return [](ov::Node* node) -> bool {
        return ov::is_type<v0::Constant>(node) || ov::is_type<v0::Parameter>(node) || ov::is_type<op_util::ShapeOfBase>(node);
    };
}

}  // namespace

DisableFP16CompFlux2RoPEPattern::DisableFP16CompFlux2RoPEPattern() {
    using namespace ov::pass::pattern;

    // This pass runs *before* RoPEFusion, so the RoPE is still decomposed and the
    // fused ov::op::internal::RoPE op does not exist yet. We therefore anchor on the
    // decomposed application graph, mirroring ov::pass::RoPEFusionFlux but without the
    // symbolic shape constraints (there is no symbolic inference context here):
    //
    //   x1        = Reshape(x)
    //   x1_0,x1_1 = Split(x1, axis=-1, num_splits=2)
    //   x2        = Concat(-x1_1, x1_0, axis=-1)      // rotate_half(x)
    //   x3        = Reshape(x2)
    //   y         = x * t_cos + x3 * t_sin            // <-- match root
    //
    // Only t_cos / t_sin captured from this structure are genuine rotary tables, so we
    // walk backward from exactly those two tensors. This avoids pulling unrelated
    // MatMul->Cos / Concat->Cos subgraphs into FP32.
    auto x = any_input();
    auto t_cos = any_input();
    auto t_sin = any_input();

    auto x1 = wrap_type<v1::Reshape>({x, any_input()});
    auto split = wrap_type<v1::Split>({x1, -1}, {{"num_splits", 2}});
    split->set_output_size(2);

    // The negation of one split half is optionally wrapped in Squeeze/Unsqueeze
    // depending on which transformations ran before this pass (see RoPEFusionFlux).
    auto opt_squeeze = optional<v0::Squeeze>({split->output(1), -1});
    auto x1_1_neg = wrap_type<v1::Multiply>({opt_squeeze, -1}, {{"auto_broadcast", "numpy"}});
    auto opt_squeeze_1 = optional<v0::Squeeze>({x1_1_neg, -1});
    auto opt_unsqueeze = optional<v0::Unsqueeze>({opt_squeeze_1, -1});

    auto x2 = wrap_type<v0::Concat>({opt_unsqueeze, split->output(0)}, {{"axis", -1}});
    auto x3 = wrap_type<v1::Reshape>({x2, any_input()});

    auto y1 = wrap_type<v1::Multiply>({x, t_cos}, {{"auto_broadcast", "numpy"}});
    auto y2 = wrap_type<v1::Multiply>({x3, t_sin}, {{"auto_broadcast", "numpy"}});
    auto result = wrap_type<v1::Add>({y1, y2}, {{"auto_broadcast", "numpy"}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root()))
            return false;
        const auto& pattern_map = m.get_pattern_value_map();
        const auto skip_pred = skip_node_predicate();
        mark_path_from_output(pattern_map.at(t_cos).get_node(), m_visited, skip_pred);
        mark_path_from_output(pattern_map.at(t_sin).get_node(), m_visited, skip_pred);
        return false;
    };

    register_matcher(std::make_shared<Matcher>(result, "DisableFP16CompFlux2RoPEPattern"), callback);
}

}  // namespace ov::intel_gpu
