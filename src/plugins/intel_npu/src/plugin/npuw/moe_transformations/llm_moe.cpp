// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_moe.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../logging.hpp"
#include "../partitioning/patterns/fold_const.hpp"
#include "../partitioning/patterns/moe.hpp"
#include "apply_moe_device_routed_transforms.hpp"
#include "moe_topology.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"
#include "openvino/util/common_util.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"
#include "transformations/op_conversions/convert_gather_matmul_to_compressed.hpp"

namespace {

// MoeOpFusion replaces the matched tiled-MoE block with a MOE(Compressed) node whose output rank
// ([B,S,H]) can differ from the rank the original (unfused) block produced (e.g. flattened
// [tokens,H]). Leftover Reshape ops right after the block - which existed to restore [B,S,H] from
// that original rank and may carry stale/baked target values for it - no longer agree with the new
// rank once ReshapeToStatic turns shapes concrete, and fail shape validation. Collapse such a
// Reshape->Reshape pair into just the second Reshape: when its special_zero is false its output
// shape depends only on its own target (not on the intermediate shape), so this is safe regardless
// of the first reshape's target values, as long as it has no other consumers.
class CollapseRedundantReshapePair : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("CollapseRedundantReshapePair");
    CollapseRedundantReshapePair() {
        using namespace ov::pass::pattern;
        auto reshape_input = any_input();
        auto reshape_a_target = any_input();
        auto reshape_a = wrap_type<ov::op::v1::Reshape>({reshape_input, reshape_a_target}, consumers_count(1));
        auto reshape_b_target = any_input();
        auto reshape_b = wrap_type<ov::op::v1::Reshape>({reshape_a, reshape_b_target});

        ov::matcher_pass_callback callback = [=](Matcher& m) {
            const auto& pm = m.get_pattern_value_map();
            auto reshape_b_node = ov::as_type_ptr<ov::op::v1::Reshape>(m.get_match_root());
            if (!reshape_b_node || reshape_b_node->get_special_zero()) {
                return false;
            }
            auto reshape_a_node = pm.at(reshape_a).get_node_shared_ptr();
            auto new_reshape =
                std::make_shared<ov::op::v1::Reshape>(pm.at(reshape_input), pm.at(reshape_b_target), false);
            new_reshape->set_friendly_name(reshape_b_node->get_friendly_name());
            ov::copy_runtime_info({reshape_a_node, reshape_b_node}, new_reshape);
            ov::replace_node(reshape_b_node, new_reshape);
            return true;
        };
        auto m = std::make_shared<Matcher>(reshape_b, "CollapseRedundantReshapePair");
        register_matcher(m, callback);
    }
};

// The fusion re-creates weight Constants as buffer-sharing reshapes without their weightless cache
// attribute, which makes NPUW deep-copy every one of them. Same buffer means same file offset, so
// put the attribute back on Constants which share a buffer with an original one.
struct OriginalWeightAttr {
    ov::WeightlessCacheAttribute attr;
    std::size_t byte_size;
};

std::unordered_map<const void*, OriginalWeightAttr> collect_weightless_attrs(const std::shared_ptr<ov::Model>& model) {
    std::unordered_map<const void*, OriginalWeightAttr> attrs;
    for (const auto& op : model->get_ops()) {
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!c) {
            continue;
        }
        const auto& rt_info = c->get_rt_info();
        auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
        if (it != rt_info.end()) {
            attrs.emplace(c->get_data_ptr(),
                          OriginalWeightAttr{it->second.as<ov::WeightlessCacheAttribute>(), c->get_byte_size()});
        }
    }
    return attrs;
}

void restore_weightless_attrs(const std::shared_ptr<ov::Model>& model,
                              const std::unordered_map<const void*, OriginalWeightAttr>& attrs) {
    for (const auto& op : model->get_ops()) {
        auto c = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!c) {
            continue;
        }
        auto& rt_info = c->get_rt_info();
        if (rt_info.count(ov::WeightlessCacheAttribute::get_type_info_static())) {
            continue;
        }
        auto it = attrs.find(c->get_data_ptr());
        if (it != attrs.end() && it->second.byte_size == c->get_byte_size()) {
            rt_info[ov::WeightlessCacheAttribute::get_type_info_static()] = it->second.attr;
        }
    }
}

// MoeOpFusion only builds MOECompressed, so its absence means the fusion did not happen.
void validate_moe_fused(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    if (std::none_of(ops.begin(), ops.end(), [](const auto& op) {
            return ov::is_type<ov::op::internal::MOECompressed>(op);
        })) {
        OPENVINO_THROW("MoE hint FUSED: no expert block was fused into a MOECompressed op.");
    }
}

}  // namespace

namespace ov::npuw::moe {

using ::intel_npu::npuw::llm::MoEHint;

// Prefer structural detection. Keep names as a legacy hint for older exports
// that still need normalization before their batched experts can be matched.
bool is_moe_model(const std::shared_ptr<ov::Model>& model) {
    const BatchedMoEPattern pattern;
    for (const auto& op : model->get_ops()) {
        if (pattern.match(op)) {
            LOG_INFO("Detected MoE model: supported batched expert topology");
            return true;
        }
        const std::string& node_name = op->get_friendly_name();
        // Check for MoE-specific patterns:
        // - layers.*.mlp.router (router network for expert selection)
        // - layers.*.mlp.experts (expert networks)
        // Note: "expert" also matches "experts" (plural)
        if (node_name.find(ov::npuw::patterns::moe::MLP_ROUTER_NAME) != std::string::npos ||
            node_name.find(ov::npuw::patterns::moe::MLP_EXPERT_NAME) != std::string::npos) {
            LOG_INFO("Detected MoE model: found node with MoE pattern - " << node_name);
            return true;
        }
    }
    LOG_DEBUG("Non-MoE model detected: no supported topology or legacy router/expert names");
    return false;
}

namespace {

const char* stage_name(Stage stage) {
    return stage == Stage::PREFILL ? "PREFILL" : "GENERATE";
}

void config_host_routed(ov::AnyMap& stage_config, Stage stage) {
    LOG_INFO("MoE config for " << stage_name(stage) << " stage: HOST_ROUTED (host-side expert routing)");
    const ov::AnyMap expert_opts = {
        {"NPUW_ONLINE_PIPELINE", "REP"},
        {"NPUW_ONLINE_KEEP_BLOCK_SIZE", "4"},
        {"NPUW_UNFOLD_IREQS", "NO"},
    };
    for (const auto& [key, value] : expert_opts) {
        stage_config[key] = value;
    }
    // Keep isolated experts even below the generic repeated-block threshold.
    const auto keep_key = "NPUW_ONLINE_KEEP_BLOCKS_TAGGED";
    auto keep = stage_config.find(keep_key);
    if (keep == stage_config.end() || keep->second.as<std::string>().empty()) {
        stage_config[keep_key] = "expert";
    } else {
        const auto& value = keep->second.as<std::string>();
        const auto tags = ov::util::split(value);
        if (std::find(tags.begin(), tags.end(), "expert") == tags.end()) {
            keep->second = value + ",expert";
        }
    }
    // Append "MOE" to any existing preset (e.g. "ATTN") instead of overwriting it.
    auto isol_it = stage_config.find("NPUW_ONLINE_ISOLATE");
    if (isol_it != stage_config.end() && !isol_it->second.as<std::string>().empty()) {
        isol_it->second = isol_it->second.as<std::string>() + ",MOE";
        LOG_INFO("MoE config: appended MOE to NPUW_ONLINE_ISOLATE -> " << isol_it->second.as<std::string>());
    } else {
        stage_config["NPUW_ONLINE_ISOLATE"] = "MOE";
    }
}

void config_device_routed(ov::AnyMap& stage_config, Stage stage) {
    if (stage == Stage::PREFILL) {
        OPENVINO_THROW("MoE DEVICE_ROUTED is not supported for PREFILL stage. "
                       "DEVICE_ROUTED mode uses in-graph gather-based expert selection which is only "
                       "optimized for GENERATE stage. Please use HOST_ROUTED or DENSE for PREFILL.");
    }
    stage_config["NPUW_UNFOLD_IREQS"] = "YES";
}

void config_dense(ov::AnyMap& stage_config, Stage stage) {
    LOG_INFO("MoE config for " << stage_name(stage) << " stage: DENSE (all experts active)");
    // DENSE mode requires CPU-only device due to extremely long NPU compilation time and high resource consumption
    auto npuw_devices = stage_config.count("NPUW_DEVICES") ? stage_config.at("NPUW_DEVICES").as<std::string>() : "NPU";
    OPENVINO_ASSERT(npuw_devices == "CPU",
                    "MoE DENSE mode requires CPU-only device (NPUW_DEVICES must be 'CPU'). "
                    "DENSE activates all experts simultaneously, causing extremely long NPU compilation time. "
                    "Please set NPUW_DEVICES to 'CPU'.");
}

void config_fused(ov::AnyMap& stage_config, Stage stage) {
    // The expert subgraph was already collapsed into a single MOECompressed op earlier
    // (see apply_moe_fusion), so no per-expert partitioning hints are needed here.
    LOG_INFO("MoE config for " << stage_name(stage)
                               << " stage: FUSED (single MOECompressed op, device dynamic-sparse kernel)");
    // The accuracy reference device cannot run the fused MOECompressed op (no CPU implementation).
    if (auto it = stage_config.find("NPUW_ACC_CHECK"); it != stage_config.end()) {
        const auto value = it->second.as<std::string>();
        if (value == "YES" || value == "true" || value == "1") {
            OPENVINO_THROW("NPUW_ACC_CHECK is not supported with MoE hint FUSED (",
                           stage_name(stage),
                           " stage): the reference device cannot run the fused MOECompressed op. "
                           "Disable NPUW_ACC_CHECK or use HOST_ROUTED or DEVICE_ROUTED.");
        }
    }
    if (stage == Stage::GENERATE) {
        stage_config["NPUW_UNFOLD_IREQS"] = "YES";
    }
}

}  // namespace

void apply_moe_config(ov::AnyMap& stage_config, MoEHint moe_hint, Stage stage) {
    switch (moe_hint) {
    case MoEHint::HOST_ROUTED:
        config_host_routed(stage_config, stage);
        return;
    case MoEHint::DEVICE_ROUTED:
        config_device_routed(stage_config, stage);
        return;
    case MoEHint::DENSE:
        config_dense(stage_config, stage);
        return;
    case MoEHint::FUSED:
        config_fused(stage_config, stage);
        return;
    }
    OPENVINO_THROW("Unknown MoE hint for ", stage_name(stage), " stage");
}

void apply_moe_fusion(const std::shared_ptr<ov::Model>& model) {
    LOG_INFO("Fusing MoE expert subgraphs into MOECompressed ops (MoEHint::FUSED)");
    const std::vector<ov::element::Type> supported_weights_types{ov::element::u4,
                                                                 ov::element::i4,
                                                                 ov::element::nf4,
                                                                 ov::element::u8,
                                                                 ov::element::i8};
    const std::vector<ov::element::Type> supported_activation_types{ov::element::f32, ov::element::f16};

    const auto original_attrs = collect_weightless_attrs(model);

    ov::pass::Manager manager("NPUW:MoEFusion");
    manager.register_pass<ov::pass::ConvertTiledMoeBlockToGatherMatmuls>(supported_weights_types);
    manager.register_pass<ov::pass::Validate>();
    // No SupportsPredicate yet: accept every matched config until NPU kernel limits are finalized
    // with the kernel team, then gate IC/OC/group_size here.
    manager.register_pass<ov::pass::ConvertGatherMatmulToGatherMatmulCompressed>(supported_activation_types,
                                                                                 supported_weights_types);
    manager.register_pass<ov::pass::MoeOpFusion>();
    manager.register_pass<CollapseRedundantReshapePair>();
    manager.register_pass<ov::pass::Validate>();
    manager.run_passes(model);
    validate_moe_fused(model);
    restore_weightless_attrs(model, original_attrs);
}

bool can_device_route_model(const std::shared_ptr<ov::Model>& model) {
    bool found = false;
    const BatchedMoEPattern pattern;
    for (const auto& node : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v1::ReduceSum>(node)) {
            continue;
        }
        ov::pass::pattern::Matcher matcher(pattern.root(), "LLMMoEDeviceEligibility");
        if (matcher.match(node)) {
            const auto topology = pattern.extract(matcher);
            if (!topology || !can_device_route(*topology)) {
                return false;
            }
            found = true;
        }
    }
    return found;
}

void fuse_requested_stages(const ::intel_npu::Config& cfg,
                           const std::shared_ptr<ov::Model>& prefill_model,
                           const std::shared_ptr<ov::Model>& generate_model) {
    // PREFILL and GENERATE hints are independent, so each stage is fused (or not) on its own.
    if (cfg.get<::intel_npu::NPUW_LLM_PREFILL_MOE_HINT>() == MoEHint::FUSED) {
        apply_moe_fusion(prefill_model);
    }
    if (cfg.get<::intel_npu::NPUW_LLM_GENERATE_MOE_HINT>() == MoEHint::FUSED) {
        apply_moe_fusion(generate_model);
    }
}

void configure_static_stages(::intel_npu::Config& cfg,
                             bool allow_auto_device_routed,
                             const std::shared_ptr<ov::Model>& prefill_model,
                             const std::vector<std::shared_ptr<ov::Model>>& generate_variants,
                             ov::AnyMap& prefill_config,
                             ov::AnyMap& generate_config) {
    // 1. Fold shape-compute chains (ShapeOf->Gather->Concat etc.) before online partitioning runs
    // pattern matching (e.g. GPTOSSRouter). Needs static shapes so ShapeOf bounds are resolvable.
    ov::npuw::patterns::util::FoldShapeComputeChain().run_on_model(prefill_model);
    for (const auto& variant : generate_variants) {
        ov::npuw::patterns::util::FoldShapeComputeChain().run_on_model(variant);
    }

    // 2. Resolve the per-stage hints. DEVICE_ROUTED may be picked for GENERATE automatically.
    const auto prefill_hint = cfg.get<::intel_npu::NPUW_LLM_PREFILL_MOE_HINT>();
    auto generate_hint = cfg.get<::intel_npu::NPUW_LLM_GENERATE_MOE_HINT>();
    if (allow_auto_device_routed && !generate_variants.empty() &&
        std::all_of(generate_variants.begin(), generate_variants.end(), can_device_route_model)) {
        generate_hint = MoEHint::DEVICE_ROUTED;
        cfg.update({{"NPUW_LLM_GENERATE_MOE_HINT", "DEVICE_ROUTED"}});
    }

    // 3. Apply each stage's config.
    apply_moe_config(prefill_config, prefill_hint, Stage::PREFILL);
    apply_moe_config(generate_config, generate_hint, Stage::GENERATE);

    // 4. DEVICE_ROUTED also rewrites the generate models (PREFILL does not support it).
    if (generate_hint == MoEHint::DEVICE_ROUTED) {
        for (const auto& variant : generate_variants) {
            ov::npuw::ApplyMoEDeviceRoutedTransforms().run_on_model(variant);
        }
    }
}

}  // namespace ov::npuw::moe
