// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/moe.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/moe_gather.hpp>
#include <intel_gpu/primitives/moe_scatter_reduction.hpp>
#include <intel_gpu/primitives/swiglu.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

#include <pugixml.hpp>
#include "intel_gpu/op/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/op/moe_compressed.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/primitives/moe_gemm.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace op {
namespace internal {
using MOE3GemmFusedCompressed = ov::intel_gpu::op::MOE3GemmFusedCompressed;
using MOECompressed = ov::intel_gpu::op::MOECompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov::intel_gpu {
using namespace cldnn;

static void CreateMOE3GemmFusedCompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::intel_gpu::op::MOE3GemmFusedCompressed>& op) {
    using input_idx = cldnn::moe_3gemm_fused_compressed::input_index;
    auto inputs = p.GetInputInfo(op);
    const auto& config = op->get_config();
    const auto& model = p.get_model();
    std::string weights_path;
    const size_t lru_expert_num = p.get_config().get_moe_offload_max_experts();
    const bool otd_enabled = lru_expert_num > 0;
    if (otd_enabled) {
        const auto& rt = model->get_rt_info();
        auto it = rt.find("__weights_path");
        OPENVINO_ASSERT(it != rt.end(), "Model rt_info is missing '__weights_path' required by OTD");
        weights_path = it->second.as<std::string>();
    }

    struct XmlConstEntry {
        size_t offset = 0;
        size_t size = 0;
        bool used = false;
    };

    std::unordered_map<std::string, std::vector<XmlConstEntry>> xml_const_entries_by_name;
    bool xml_offsets_ready = false;

    auto load_const_offsets_from_xml = [&]() {
        if (xml_offsets_ready || weights_path.empty()) {
            return;
        }

        std::filesystem::path xml_path(weights_path);
        xml_path.replace_extension(".xml");
        OPENVINO_ASSERT(std::filesystem::exists(xml_path), "IR xml file is not found: ", xml_path.string());

        pugi::xml_document doc;
        OPENVINO_ASSERT(doc.load_file(xml_path.string().c_str()), "Failed to parse IR xml file: ", xml_path.string());

        auto net = doc.child("net");
        auto layers = net.child("layers");
        for (auto layer = layers.child("layer"); layer; layer = layer.next_sibling("layer")) {
            const auto type_attr = layer.attribute("type");
            if (!type_attr || std::string(type_attr.value()) != "Const") {
                continue;
            }

            const auto data = layer.child("data");
            const auto name_attr = layer.attribute("name");
            const auto offset_attr = data.attribute("offset");
            const auto size_attr = data.attribute("size");
            if (!data || !name_attr || !offset_attr || !size_attr) {
                continue;
            }

            XmlConstEntry entry;
            entry.offset = static_cast<size_t>(std::stoull(offset_attr.value()));
            entry.size = static_cast<size_t>(std::stoull(size_attr.value()));
            xml_const_entries_by_name[name_attr.value()].push_back(entry);
        }

        xml_offsets_ready = true;
    };

    auto get_const_offset = [&](size_t index) -> size_t {
        auto node = op->input_value(index).get_node_shared_ptr();
        auto const_op = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
        OPENVINO_ASSERT(const_op != nullptr, "Expected constant input for MOE3GemmFusedCompressed");
        const auto& rt_info = const_op->get_rt_info();
        auto attr_it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());

        if (attr_it != rt_info.end()) {
            return attr_it->second.as<ov::WeightlessCacheAttribute>().bin_offset;
        }

        load_const_offsets_from_xml();
        OPENVINO_ASSERT(xml_offsets_ready,
                        "Missing WeightlessCacheAttribute and failed to initialize xml-based offset lookup for "
                        "MOE3GemmFusedCompressed constant input");

        auto resolve_from_name = [&](const std::string& lookup_name,
                                     const std::string& const_name,
                                     size_t expected_size,
                                     size_t& resolved_offset) -> bool {
            auto by_name_it = xml_const_entries_by_name.find(lookup_name);
            if (by_name_it == xml_const_entries_by_name.end()) {
                return false;
            }

            size_t match_count = 0;
            XmlConstEntry* matched_entry = nullptr;
            for (auto& entry : by_name_it->second) {
                if (!entry.used && entry.size == expected_size) {
                    match_count++;
                    if (matched_entry == nullptr) {
                        matched_entry = &entry;
                    }
                }
            }

            if (match_count == 1 && matched_entry != nullptr) {
                matched_entry->used = true;
                resolved_offset = matched_entry->offset;
                return true;
            }

            if (match_count > 1) {
                OPENVINO_THROW("Ambiguous xml offset resolution for MOE3GemmFusedCompressed constant input: ",
                               const_name,
                               ", lookup_name=", lookup_name,
                               ", byte_size=", expected_size,
                               ", candidates=", match_count);
            }

            return false;
        };

        const auto& name = const_op->get_friendly_name();
        const size_t expected_size = const_op->get_byte_size();
        size_t resolved_offset = 0;
        if (resolve_from_name(name, name, expected_size, resolved_offset)) {
            return resolved_offset;
        }

        // Try original/fused names before using any size-based fallback.
        std::set<std::string> fused_names_unique;
        for (const auto& fused_name : ov::getFusedNamesVector(const_op)) {
            if (!fused_name.empty() && fused_name != name) {
                fused_names_unique.insert(fused_name);
            }
        }
        for (const auto& fused_name : fused_names_unique) {
            if (resolve_from_name(fused_name, name, expected_size, resolved_offset)) {
                return resolved_offset;
            }
        }

        // Fallback: allow by-size only when there is exactly one unused candidate.
        struct SizeCandidate {
            std::string name;
            XmlConstEntry* entry = nullptr;
        };
        std::vector<SizeCandidate> size_candidates;
        for (auto& kv : xml_const_entries_by_name) {
            for (auto& entry : kv.second) {
                if (!entry.used && entry.size == expected_size) {
                    size_candidates.push_back(SizeCandidate{kv.first, &entry});
                }
            }
        }

        if (size_candidates.size() == 1 && size_candidates[0].entry != nullptr) {
            size_candidates[0].entry->used = true;
            return size_candidates[0].entry->offset;
        }

        if (size_candidates.size() > 1) {
            std::ostringstream oss;
            const size_t max_candidates_to_log = 8;
            for (size_t i = 0; i < std::min(max_candidates_to_log, size_candidates.size()); i++) {
                const auto* candidate_entry = size_candidates[i].entry;
                if (candidate_entry == nullptr) {
                    continue;
                }
                if (i > 0) {
                    oss << ';';
                }
                oss << size_candidates[i].name << '@' << candidate_entry->offset;
            }

            OPENVINO_THROW("Ambiguous xml offset resolution for MOE3GemmFusedCompressed constant input: ",
                           name,
                           ", byte_size=", expected_size,
                           ", size_candidates=", size_candidates.size(),
                           ", sample_candidates=", oss.str());
        }

        OPENVINO_THROW("Unable to resolve xml offset for MOE3GemmFusedCompressed constant input: ", name,
                       ", byte_size=", expected_size);
    };

    const std::array<size_t, cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count> const_input_idx_by_offset = {
        static_cast<size_t>(input_idx::weight_0),
        static_cast<size_t>(input_idx::weight_1),
        static_cast<size_t>(input_idx::weight_2),
        static_cast<size_t>(input_idx::scale_0),
        static_cast<size_t>(input_idx::scale_1),
        static_cast<size_t>(input_idx::scale_2),
        static_cast<size_t>(input_idx::zp_0),
        static_cast<size_t>(input_idx::zp_1),
        static_cast<size_t>(input_idx::zp_2)
    };

    std::vector<size_t> weight_bin_offsets(cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count, 0);
    // Serialized offsets are only needed for OTD path (weight-on-demand loading).
    if (otd_enabled) {
        for (size_t i = 0; i < const_input_idx_by_offset.size(); i++) {
            weight_bin_offsets[i] = get_const_offset(const_input_idx_by_offset[i]);
        }
    }
    ///   0: hidden_states - input tensor with hidden representations
    ///   1: routing_weights - [num_seq, num_experts] routing weights for all experts
    ///   2: w0_weight - expert weights for first projection,
    ///                  shape [num_experts, inter_size, group_num, group_size]
    ///   3: w0_scale - expert scale for first projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   4: w0_zp - expert zp for first projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   5: w1_weight - expert weights for second projection,
    ///                  shape [num_experts, inter_size, group_num, group_size]
    ///   6: w1_scale - expert scale for second projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   7: w1_zp - expert zp for second projection for compressed experts,
    ///                  shape [num_experts, inter_size, group_num, 1]
    ///   8: w2_weight - expert weights for final projection,
    ///                  shape [num_experts, hidden_size, group_num, group_size]
    ///   9: w2_scale - expert scale for final projection for compressed experts,
    ///                  shape [num_experts, hidden_size, group_num, 1]
    ///   10: w2_zp - expert zp for final projection for compressed experts,
    ///                  shape [num_experts, hidden_size, group_num, 1]
    ///   11: routing_bias (optional, SIGMOID_BIAS only; dummy placeholder for SOFTMAX+shared) -
    ///                  [1, num_experts] routing bias for sigmoid routing
    ///   12: routing_eps (optional, SIGMOID_BIAS only; dummy placeholder for SOFTMAX+shared) -
    ///                  scalar epsilon for normalization
    ///
    ///   Options for shared experts (if config.num_shared_expert > 0, always starting at index 13):
    ///   13: shared_gate_weight - shared expert weights for first projection,
    ///                   shape [1, inter_size, group_num, group_size]
    ///   14: shared_gate_scale - shared expert scale for first projection,
    ///                   shape [1, inter_size, group_num, 1]
    ///   15: shared_gate_zp - shared expert zp for first projection,
    ///                   shape [1, inter_size, group_num, 1]
    ///   16: shared_up_weight - shared expert weights for second projection,
    ///                   shape [1, inter_size, group_num, group_size]
    ///   17: shared_up_scale - shared expert scale for second projection,
    ///                   shape [1, inter_size, group_num, 1]
    ///   18: shared_up_zp - shared expert zp for second projection,
    ///                   shape [1, inter_size, group_num, 1]
    ///   19: shared_down_weight - shared expert weights for final projection,
    ///                   shape [1, hidden_size, group_num, group_size]
    ///   20: shared_down_scale - shared expert scale for final projection,
    ///                   shape [1, hidden_size, group_num, 1]
    ///   21: shared_down_zp - shared expert zp for final projection,
    ///                   shape [1, hidden_size, group_num, 1]
    ///   22: shared_gate_gate_weight - shared expert gate weight for gating,
    ///                   shape [hidden_size]
    const size_t expected_inputs = config.num_shared_expert > 0 ? 23
                                 : config.routing_type == op::MOECompressed::RoutingType::SIGMOID_BIAS ? 13
                                 : 11;
    validate_inputs_count(op, {expected_inputs});

    const std::string layerName = layer_type_name_ID(op);
    const cldnn::moe_3gemm_fused_compressed moe(layerName, inputs, config, weight_bin_offsets, weights_path, lru_expert_num);

    p.add_primitive(*op, moe);
}

static void CreateMOECompressedOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::MOECompressed>& op) {
    auto inputs = p.GetInputInfo(op);
    auto& config = op->get_config();
    std::vector<cldnn::input_info> input_infos;
    for (const auto& input : inputs) {
        input_infos.push_back(cldnn::input_info(input));
    }
    if (config.expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
        // Create GEMM3_SWIGLU specific primitives
        //   0: hidden_states - input tensor with hidden representations
        //   1: routing_weights - [num_experts, ...] normalized weights for selected experts
        //      (input to final multiplication)
        //   2: router_topk_output_indices - [..., topk] indices of selected top-k experts
        //   3: w0_weight - expert weights for first projection,
        //   shape [num_experts, inter_size, group_num, group_size]
        //   4: w0_scale - expert scale for first projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   5: w0_zp - expert zp for first projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   6: w1_weight - expert weights for second projection,
        //   shape [num_experts, inter_size, group_num, group_size]
        //   7: w1_scale - expert scale for second projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   8: w1_zp - expert zp for second projection for compressed experts,
        //   shape [num_experts, inter_size, group_num, 1]
        //   9: w2_weight - expert weights for final projection,
        //   shape [num_experts, hidden_size, group_num, group_size]
        //   10: w2_scale - expert scale for final projection for compressed experts,
        //   shape [num_experts, hidden_size, group_num, 1]
        //   11: w2_zp - expert zp for final projection for compressed experts,
        //   shape [num_experts, hidden_size, group_num, 1]

        // Use moe_3gemm_fused_compressed to replace it.
    } else {
        // Create GEMM2_BIAS_SWIGLU_CLAMP specific primitives
        // input0 : input {#tokens, hidden_size}
        // input1 : topk_weight {#tokens, num_experts_per_token}
        // input2 : topk_idx {#tokens, num_experts_per_token}
        // input3 : compressed_weights_input_up {#experts, ofm, num_groups, group_size}
        // input4 : scale_input_up {#experts, ofm, num_groups, 1}
        // input5 : bias_up {#experts, 1, ofm}
        // input6 : compressed_weights_input_down {#experts, ofm, num_groups, group_size}
        // input7 : scale_input_down {#experts, ofm, num_groups, 1}
        // input8 : bias_down {#experts, 1, ofm}
        // moe_mask_gen
        // moe_gather
        // moe_gemm_up + bias
        // swiglu_with_clamp
        // moe_gemm_down + bias
        // moe_scatter_reduce
        std::string prim_name_base = layer_type_name_ID(op);
        auto moe_mask_gen_name = prim_name_base + "_moe_mask_gen";
        auto moe_mask_gen_reshape_name = prim_name_base + "_moe_mask_gen_reshape";
        auto moe_gather_name = prim_name_base + "_moe_gather";
        auto moe_bias_up_name = prim_name_base + "_moe_bias_up";
        auto moe_gemm_up_name = prim_name_base + "_moe_gemm_up";
        auto moe_swiglu_name = prim_name_base + "_moe_swiglu";
        auto moe_gemm_down_name = prim_name_base + "_moe_gemm_down";
        auto moe_bias_down_name = prim_name_base + "_moe_bias_down";
        auto moe_scatter_reduce_name = prim_name_base + "_moe_scatter_reduce";
        auto moe_mask_gen_prim = cldnn::moe_mask_gen(moe_mask_gen_name,
                                                     input_infos[2],  // topk indices
                                                     static_cast<int32_t>(config.num_expert),
                                                     static_cast<int32_t>(config.top_k),
                                                     true);
        p.add_primitive(*op, moe_mask_gen_prim);
        auto moe_mask_gen_reshape_prim =
            cldnn::moe_mask_gen_reshape(moe_mask_gen_reshape_name,
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::TOKENS_PER_EXPERT),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::EXPERTS_INFO_START_IDX),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::EXPERTS_ID),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::TOKENS_LENS_PER_EXPERT),
                                        input_info(moe_mask_gen_prim, moe_mask_gen::MoEMaskGenOutputIdx::NUM_ACTUALLY_USED_EXPERTS));
        p.add_primitive(*op, moe_mask_gen_reshape_prim);
        auto moe_gather_prim = cldnn::moe_gather(moe_gather_name,
                                                 input_infos[0],  // input
                                                 input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                                                 config);
        p.add_primitive(*op, moe_gather_prim);
        std::vector<cldnn::input_info> moe_gemm_up_inputs = {
            input_info(moe_gather_name),  // topk_weight
            input_infos[3],               // compressed_weights_input_up
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT)};
        size_t down_idx = 0;
        if (config.has_zp) {
            moe_gemm_up_inputs.push_back(input_infos[6]);  // bias_up
            moe_gemm_up_inputs.push_back(input_infos[4]);  // scale_input_up
            moe_gemm_up_inputs.push_back(input_infos[5]);  // zp_input_up
            down_idx = 7;
        } else {
            moe_gemm_up_inputs.push_back(input_infos[5]);  // bias_up
            moe_gemm_up_inputs.push_back(input_infos[4]);  // scale_input_up
            down_idx = 6;
        }
        auto moe_gemm_up = cldnn::moe_gemm(moe_gemm_up_name, moe_gemm_up_inputs, config);
        moe_gemm_up.has_bias = true;
        p.add_primitive(*op, moe_gemm_up);

        // gpt-oss swiglu pattern
        // config.expert_alpha : clamp_max
        // config.expert_beta : swish_beta which is slightly different from usual swiglu pattern
        // - Applied clamp
        // - Added one for up value
        // - Gate stride is 1 (not splitting to half and half)
        // - config.expert_alpha : clamp_max
        // - config.expert_beta : swish_beta
        // TODO : update for each new pattern
        auto moe_swiglu_prim = cldnn::swiglu(moe_swiglu_name,
                                             input_info(moe_gemm_up_name),
                                             2,  // axis
                                             2,  // glu_stride
                                             ov::op::internal::GLU::GluType::Swish,
                                             0,                     // gate idx
                                             -config.expert_alpha,  // clamp_min
                                             config.expert_alpha,   // clamp_max
                                             config.expert_beta,    // swish beta
                                             1.0f,                  // up_add_val
                                             cldnn::tensor());
        p.add_primitive(*op, moe_swiglu_prim);
        std::vector<cldnn::input_info> moe_gemm_down_inputs = {
            input_info(moe_swiglu_name),
            input_infos[down_idx],  // compressed_weights_input_down
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
            input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
        };

        if (config.has_zp) {
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 3]);  // bias_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 1]);  // scale_input_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 2]);  // zp_input_up
        } else {
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 2]);  // bias_up
            moe_gemm_down_inputs.push_back(input_infos[down_idx + 1]);  // scale_input_up
        }

        auto moe_gemm_down = cldnn::moe_gemm(moe_gemm_down_name, moe_gemm_down_inputs, config);
        moe_gemm_down.has_bias = true;
        p.add_primitive(*op, moe_gemm_down);
        auto moe_scatter_reduce_prim =
            cldnn::moe_scatter_reduction(moe_scatter_reduce_name,
                                         input_info(moe_gemm_down_name),
                                         input_infos[2],
                                         input_infos[1],
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_PER_EXPERT),
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_INFO_START_IDX),
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::TOKENS_LENS_PER_EXPERT),
                                         input_info(moe_mask_gen_reshape_name, moe_mask_gen_reshape::MoEMaskGenReshapeOutputIdx::EXPERTS_ID),
                                         config,
                                         true);
        p.add_primitive(*op, moe_scatter_reduce_prim);
    }
}
REGISTER_FACTORY_IMPL(internal, MOE3GemmFusedCompressed);
REGISTER_FACTORY_IMPL(internal, MOECompressed);

}  // namespace ov::intel_gpu
