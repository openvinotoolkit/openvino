// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_rms_input_precision.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/symbolic_transformations/symbolic_optimizations.hpp"
#include "transformations/utils/utils.hpp"
#include "utils.hpp"

namespace ov::intel_gpu {

IncreaseRMSInputPrecision::IncreaseRMSInputPrecision() {}

bool IncreaseRMSInputPrecision::run_on_model(const std::shared_ptr<ov::Model>& model) {
    using namespace ov::pass;
    SymbolicOptimizations symbolic_optimizations(false, get_pass_config());
    auto symbolic_ctx_manager = symbolic_optimizations.get_manager();
    symbolic_ctx_manager->register_pass<IncreasePrecisionForQwenVLMerger>();
    symbolic_ctx_manager->register_pass<Validate>();
    return symbolic_optimizations.run_on_model(model);
}

IncreasePrecisionForQwenVLMerger::IncreasePrecisionForQwenVLMerger() {
    using namespace ov::op;
    using namespace ov::pass::pattern;
    using namespace ov::pass::pattern::op;

    // Pattern: Qwen2.5-VL Vision Embedings Merger last block (blocks.31) + Merger MLP
    // Target: down_proj MatMul -> Add -> aten_add -> RMS -> Reshape -> merger_mlp
    auto attn_proj = wrap_type<v0::MatMul>({any_input(), any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto attn_proj_add = wrap_type<ov::op::v1::Add>({attn_proj, any_input()}, type_matches(element::f16));
    auto attn_add = wrap_type<ov::op::v1::Add>({any_input(), attn_proj_add}, type_matches(element::f16));
    auto rms_post_m = wrap_type<ov::op::internal::RMS>({attn_add, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    auto gate_proj = wrap_type<v0::MatMul>({rms_post_m, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto up_proj = wrap_type<v0::MatMul>({rms_post_m, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto gate_proj_add = wrap_type<ov::op::v1::Add>({gate_proj, any_input()}, type_matches(element::f16));
    auto up_proj_add = wrap_type<ov::op::v1::Add>({up_proj, any_input()}, type_matches(element::f16));
    auto act_silu = wrap_type<ov::op::v4::Swish>({gate_proj_add}, type_matches(ov::element::f16) && consumers_count(1));
    auto aten_mul = wrap_type<ov::op::v1::Multiply>({act_silu, up_proj_add}, type_matches(ov::element::f16) && consumers_count(1));
    auto down_proj = wrap_type<v0::MatMul>({aten_mul, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto down_proj_add = wrap_type<ov::op::v1::Add>({down_proj, any_input()}, type_matches(element::f16));
    auto aten_add = wrap_type<v1::Add>({attn_add, down_proj_add}, type_matches(element::f16));
    auto rms_m = wrap_type<internal::RMS>({aten_add, wrap_type<v0::Constant>()}, type_matches(element::f16));
    auto aten_view = wrap_type<v1::Reshape>({rms_m, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto merger_mlp0 = wrap_type<v0::MatMul>({aten_view, any_input()}, type_matches(ov::element::f16) && consumers_count(1));
    auto merger_mlp0_add = wrap_type<v1::Add>({merger_mlp0, any_input()}, type_matches(element::f16));
    auto merger_mlp1_gelu = wrap_type<v7::Gelu>({merger_mlp0_add}, type_matches(ov::element::f16) && consumers_count(1));
    auto merger_mlp2 = wrap_type<v0::MatMul>({merger_mlp1_gelu, any_input()}, type_matches(ov::element::f16) && consumers_count(1));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto matmul = ov::as_type_ptr<v0::MatMul>(pattern_map.at(down_proj).get_node_shared_ptr());
        auto add_1 = ov::as_type_ptr<v1::Add>(pattern_map.at(down_proj_add).get_node_shared_ptr());
        auto add_2 = ov::as_type_ptr<v1::Add>(pattern_map.at(aten_add).get_node_shared_ptr());
        auto rms = ov::as_type_ptr<internal::RMS>(pattern_map.at(rms_m).get_node_shared_ptr());

        if (!rms || !add_1 || !add_2 || !matmul || transformation_callback(rms)) {
            return false;
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = matmul->get_output_element_type(0);
        if (original_et == desired_et) {
            return false;
        }

        size_t input_idx = 0;
        bool is_changed = insert_converts_before_if_needed(matmul, desired_et, input_idx);
        if (!is_changed)
            return false;
        is_changed = insert_converts_before_if_needed(add_1, desired_et, input_idx);
        if (!is_changed)
            return false;
        is_changed = insert_converts_before_if_needed(add_2, desired_et, input_idx, {1});
        if (!is_changed)
            return false;
        is_changed = insert_converts_before_if_needed(rms, desired_et, input_idx, {0});
        if (!is_changed)
            return false;

        size_t output_idx = 0;
        insert_converts_after_if_needed(rms, original_et, output_idx);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(merger_mlp2, "IncreasePrecisionForQwenVLMerger");
    this->register_matcher(m, callback);
}
}  // namespace ov::intel_gpu
