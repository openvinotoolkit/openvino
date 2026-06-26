// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_flux2_rope.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/shape_of_base.hpp"
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

    // FLUX.2 pos_embed frequency tables: outer-product MatMul -> Cos / Sin.
    auto outer_matmul = wrap_type<v0::MatMul>({any_input(), any_input()});
    auto pos_cos = wrap_type<v0::Cos>({outer_matmul});
    auto pos_sin = wrap_type<v0::Sin>({outer_matmul});

    // rotary_emb / LLM-style tables: Concat -> Cos or Sin (e.g. text encoder rotary_emb).
    auto rot_concat = wrap_type<v0::Concat>({any_input(), any_input()});
    auto table_cos = wrap_type<v0::Cos>({rot_concat});
    auto table_sin = wrap_type<v0::Sin>({rot_concat});

    // MatMul -> Transpose -> Concat -> Cos/Sin (MarkRopeInputsToKeepInMixedPrecision reference pattern).
    auto table_matmul = wrap_type<v0::MatMul>({any_input(), any_input()});
    auto table_transpose = wrap_type<v1::Transpose>({table_matmul, any_input()});
    auto table_concat = wrap_type<v0::Concat>({table_transpose, table_transpose}, {{"axis", -1}});
    auto ref_cos = wrap_type<v0::Cos>({table_concat});
    auto ref_sin = wrap_type<v0::Sin>({table_concat});

    auto register_mark_root = [this](const std::shared_ptr<ov::Node>& pattern_root, const char* name) {
        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
            if (transformation_callback(m.get_match_root()))
                return false;
            const auto skip_pred = skip_node_predicate();
            mark_path_from_output(m.get_pattern_value_map().at(pattern_root).get_node(), m_visited, skip_pred);
            return true;
        };
        register_matcher(std::make_shared<Matcher>(pattern_root, name), callback);
    };

    register_mark_root(pos_cos, "DisableFP16CompFlux2RoPEPattern_PosCos");
    register_mark_root(pos_sin, "DisableFP16CompFlux2RoPEPattern_PosSin");
    register_mark_root(table_cos, "DisableFP16CompFlux2RoPEPattern_TableCos");
    register_mark_root(table_sin, "DisableFP16CompFlux2RoPEPattern_TableSin");
    register_mark_root(ref_cos, "DisableFP16CompFlux2RoPEPattern_RefCos");
    register_mark_root(ref_sin, "DisableFP16CompFlux2RoPEPattern_RefSin");
}

}  // namespace ov::intel_gpu
