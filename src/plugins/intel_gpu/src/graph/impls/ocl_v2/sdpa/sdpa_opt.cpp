// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
// Put this file at first to avoid incorrect header files includes order.
// For example, intel_gpu/runtime/utils.hpp will causes compiling error in hash<dnnl::impl::primitive_hashing::key_t>
#include "sdpa_gen_ocl.hpp"
#include "sdpa_gen_ocl_decode.hpp"
// clang-format on

#include "sdpa_opt.hpp"

#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "kv_cache_inst.h"
#include "openvino/core/partial_shape.hpp"
#include "primitive_inst.h"
#include "scaled_dot_product_attention_inst.h"
#include "sdpa_base.hpp"
#include "sdpa_gen_opt.hpp"

using namespace cldnn;

namespace ov::intel_gpu::ocl {

namespace {

// Decode with grouped-query heads stacked as query rows has 3-4 rows per KV head on common models.
// Single-token kernel reads the keys once per row (Arc B390, 4K keys: +0.3% at 6 rows, -9% at 8, -26% at 16).
constexpr int64_t max_single_token_rows = 4;

// Several query rows stay on the multi-token kernels where the single-token stages are not exact for them:
// per-channel compressed KV needs the micro kernel, every key partition adds the sink to its softmax sum,
// the finalization stage merges at most get_max_partitions_num() partitions, and the query's local memory is K wide
// while its loads and the output reduction use the V width. A transposed V is read by the sdpa_ocl multi-token stage
// (or the one-row opt stage), never by the key-partitioned stage for several rows.
bool single_token_supports_query_rows(const RuntimeParams& params) {
    return !SDPAOpt::has_per_channel_compressed_kv(params) && !SDPABase::has_transposed_v(params) &&
           !params.typed_desc<scaled_dot_product_attention>()->has_sink_input &&
           get_partitions_num(params, SDPAStage::SINGLE_TOKEN) <= get_max_partitions_num() &&
           SDPABase::get_effective_v_head_size(params) <= SDPABase::get_effective_k_head_size(params);
}

bool use_multi_token_stage(const RuntimeParams& params) {
    const auto query_rows = get_target_seq_len(params);
    return query_rows > max_single_token_rows || (query_rows > 1 && !single_token_supports_query_rows(params));
}

// Micro single-token kernel has no key partitions, so it is used for one query row only.
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
bool single_query_row(const RuntimeParams& params) {
    return get_target_seq_len(params) <= 1;
}
#endif

}  // namespace

bool SDPAOpt::has_per_channel_compressed_kv(const kernel_impl_params& params) {
    return SDPABase::has_per_channel_compressed_kv(params);
}

bool SDPAOpt::per_channel_kv_on_dpas_lane(const kernel_impl_params& params) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    return sdpa_ocl_serves_op(params, false);
#else
    UNUSED(params);
    return false;
#endif
}

bool SDPAOpt::transposed_v_on_dpas_lane(const kernel_impl_params& params) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    return SDPABase::transposed_v_reader_ok(params) && sdpa_ocl_serves_op(params, false);
#else
    UNUSED(params);
    return false;
#endif
}

namespace {
// unaligned_head_size() of the head sizes the kernels see: a packed 4-bit per-channel cache holds two channels per byte
// (SDPABase::get_effective_k_head_size()). Every other op keeps the layouts' sizes.
bool unaligned_kv_head(const RuntimeParams& params) {
    if (!SDPABase::has_per_channel_compressed_kv(params) || !SDPABase::is_int4_kv_cache(params)) {
        return unaligned_head_size(params);
    }
    return SDPABase::get_effective_k_head_size(params) % 16 != 0 || SDPABase::get_effective_v_head_size(params) % 16 != 0;
}

// Several query rows of an unaligned head go to the multi-token stages: the key-partitioned single-token stage (use_multi_token_stage())
// does not support unaligned heads, and the sdpa_ocl single-token stage takes one row only.
bool use_multi_token_rows(const RuntimeParams& params) {
    return use_multi_token_stage(params) || (get_target_seq_len(params) > 1 && unaligned_kv_head(params));
}
}  // namespace

#ifdef ENABLE_ONEDNN_FOR_GPU
namespace {
// Knob (OV_GPU_ENABLE_SDPA_MICRO=0, a historical name): the DPAS SDPA kernel (sdpa_ocl) off, so the op runs the OCL
// sdpa_opt kernels, for accuracy A/B without rebuilding.
bool dpas_sdpa_enabled(const RuntimeParams& params) {
    if (!params.get_program().get_config().get_enable_sdpa_micro()) {
        GPU_DEBUG_TRACE_DETAIL << "[GPU] DPAS SDPA kernel (sdpa_ocl) DISABLED via OV_GPU_ENABLE_SDPA_MICRO=0; "
                                  "falling back to OCL sdpa_opt kernel.\n";
        return false;
    }
    return true;
}

// Whether the opt stages take this plain SDPA op: the head sizes with which the op stays fused on an XMX device off the
// sdpa_ocl lane (transformations_pipeline.cpp), 64 to 512 and even, and a head size that is not a multiple of 16 only
// with f16 / bf16 K and V. On the lane the op stays fused at every head size up to 256, so on xe_hpg (below) an op
// outside these keeps sdpa_ocl.
bool opt_stages_take_head(const RuntimeParams& params) {
    const auto desc = params.typed_desc<scaled_dot_product_attention>();
    const auto& k_layout = params.get_input_layout(1);
    const auto& v_layout = params.get_input_layout(2);
    // A packed 4-bit per-channel cache: the unpacked head sizes (unaligned_kv_head()).
    const bool pc_int4 = SDPABase::has_per_channel_compressed_kv(params) && SDPABase::is_int4_kv_cache(params);
    const auto k_head_size = pc_int4 ? SDPABase::get_effective_k_head_size(params) : get_head_size(k_layout, desc->input_k_transpose_order);
    const auto v_head_size = pc_int4 ? SDPABase::get_effective_v_head_size(params) : get_head_size(v_layout, desc->input_v_transpose_order);
    const auto max_head = std::min<int64_t>(static_cast<int64_t>(params.get_device_info().max_work_group_size), 512);
    for (const auto head : {k_head_size, v_head_size}) {
        if (head < 64 || head > max_head || head % 2 != 0) {
            return false;
        }
    }
    const bool f16_kv = ov::element::Type(k_layout.data_type).size() >= 2 && ov::element::Type(v_layout.data_type).size() >= 2;
    return f16_kv || !unaligned_kv_head(params);
}
}  // namespace
#endif

class SDPAOptImpl : public SDPAImplBase {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SDPAOptImpl)

    // Off as it was with the removed micro stage; replay of the remaining stages has not been checked.
    bool supports_replay() const override {
        return false;
    }

    static constexpr bool indirect = true;
    static constexpr bool prefill = true;

    Stage::Ptr indirect_single_token = make_stage<SDPAOptGeneratorSingleToken>(indirect);
    Stage::Ptr regular_single_token = make_stage<SDPAOptGeneratorSingleToken>(!indirect);

    Stage::Ptr indirect_multi_tokens = make_stage<SDPAOptGeneratorMultiToken>(indirect);
    Stage::Ptr regular_multi_tokens = make_stage<SDPAOptGeneratorMultiToken>(!indirect);

    Stage::Ptr indirect_finalization = make_stage<SDPAOptGeneratorFinalization>(indirect);
    Stage::Ptr regular_finalization = make_stage<SDPAOptGeneratorFinalization>(!indirect);

    // The DPAS kernel (sdpa_ocl). Its stages exist in every impl, because a loaded or cloned impl is default-constructed
    // and only gets _order (indices into _stages) back; the params ctor adds them where sdpa_ocl serves the op and
    // everything after that reads has_stage(), never the environment. Indices 6/7.
    // Created only with oneDNN; has_stage() of a null stage is false.
    Stage::Ptr ocl_single_token;
    Stage::Ptr ocl_multi_tokens;

    SDPAOptImpl() : SDPAImplBase(SDPAOpt::get_type_info_static()) {
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
        ocl_single_token = make_stage<SDPAOclGenerator>(!prefill);
        ocl_multi_tokens = make_stage<SDPAOclGenerator>(prefill);
#endif
    }
    explicit SDPAOptImpl(const RuntimeParams& impl_param) : SDPAOptImpl() {
        auto params = SDPABase::requires_shape_canonicalization(impl_param) ? SDPABase::static_canonicalize_shapes(impl_param) : impl_param;
        // sdpa_ocl on the lane sdpa_ocl_selected() picks (xe_hpc, Xe2+ and xe_hpg XMX; the table in docs/sdpa_ocl.md). Where it
        // is refused, and on every other device, the opt kernels run.
        const bool ocl_lane = cldnn::paged_attention::sdpa_ocl_selected(params.get_device_info());
#ifdef ENABLE_ONEDNN_FOR_GPU
        // Lazy, so only the branches below that need it evaluate the gate. sdpa_ocl reads a single-element runtime mask
        // too, and it needs no oneDNN microkernel support (the lane already means XMX).
        auto dpas_supported = [&]() {
            return SDPAOpt::sdpa_ocl_serves_op(params, true);
        };
        // xe_hpg runs a one-query step (decode) of plain SDPA on the opt stages ("Plain SDPA of stateful LLMs on xe_hpg
        // (2026-11-14)" in the docs). PREFILL keeps sdpa_ocl on every arch, and so does a decode the opt stages do not take
        // (opt_stages_take_head()), or one with a transposed V they cannot read (only the single-token stage reads it, and
        // not at a head that is not a multiple of 16: "Transposed V on xe_hpg"), which skips the ARL-H rule below too.
        const bool tv_needs_ocl_single_token = SDPABase::has_transposed_v(params) && (!SDPABase::opt_single_token_reads_transposed_v(params) ||
                                                                                      !opt_stages_take_head(params) || unaligned_kv_head(params));
        // Per-channel compressed KV likewise: the opt stages read one scale per token, but for the single-token stage on
        // xe_hpg (KV_COMP_PER_CHANNEL), at the heads it takes.
        const bool pc_needs_ocl_single_token = SDPABase::has_per_channel_compressed_kv(params) && (!SDPABase::opt_single_token_reads_per_channel(params) ||
                                                                                                   !opt_stages_take_head(params) || unaligned_kv_head(params));
        const bool needs_ocl_single_token = tv_needs_ocl_single_token || pc_needs_ocl_single_token;
        const bool ocl_single_token_ok = params.get_device_info().arch != gpu_arch::xe_hpg || !opt_stages_take_head(params) || needs_ocl_single_token;
#endif
        GPU_DEBUG_TRACE_DETAIL << "create stages for dynamic = " << params.is_dynamic() << ", sdpa_ocl lane = " << ocl_lane << "\n";
        if (params.is_dynamic()) {
            GPU_DEBUG_TRACE_DETAIL << "add stages for dynamic ...\n";
            add_stage(regular_single_token, params);
            add_stage(indirect_single_token, params);
            add_stage(regular_multi_tokens, params);
            add_stage(indirect_multi_tokens, params);
            add_stage(regular_finalization, params);
            add_stage(indirect_finalization, params);
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
            if (dpas_supported()) {
                GPU_DEBUG_TRACE_DETAIL << "add stage for sdpa_ocl dynamic ...\n";
                add_stage(ocl_multi_tokens, params);
                // A rank-3 op keeps the opt single-token kernel for one query ("Rank-3 plain SDPA" in the docs).
                if (ocl_single_token_ok && params.typed_desc<scaled_dot_product_attention>()->input_q_transpose_order.size() != 3) {
                    add_stage(ocl_single_token, params);
                }
            }
#endif
            GPU_DEBUG_TRACE_DETAIL << "add stage for dynamic done \n";
        } else {
            auto is_indirect = params.typed_desc<scaled_dot_product_attention>()->indirect_axis != -1;
            GPU_DEBUG_TRACE_DETAIL << "add stage for non-dynamic, is_indirect = " << is_indirect << "\n";
            // The sdpa_ocl single-token kernel handles unaligned head sizes (head < d / value < dv /
            // DKS_ACTIVE guards), so on the sdpa_ocl lane we let an unaligned decode fall through to the
            // single-token branch, which stages the multi-tokens kernel only if sdpa_ocl is not added.
            // The opt single-token kernel does not support unaligned head sizes, and indirect keeps
            // its opt fallback, so both are excluded.
            const bool is_prefill = use_multi_token_rows(params);
#ifdef ENABLE_ONEDNN_FOR_GPU
            const bool ocl_single_token_unaligned = ocl_lane && !is_indirect && ocl_single_token_ok;
#else
            const bool ocl_single_token_unaligned = ocl_lane && !is_indirect;
#endif
            if (is_prefill || (unaligned_kv_head(params) && !ocl_single_token_unaligned)) {
                if (is_indirect) {
                    GPU_DEBUG_TRACE_DETAIL << "add stage for indirect non-dynamic with prefill_stage \n";
                    add_stage(indirect_multi_tokens, params);
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
                } else if (is_prefill && dpas_supported()) {
                    // Prefill only: execute() never dispatches the DPAS multi-tokens stage for one query.
                    GPU_DEBUG_TRACE_DETAIL << "add stage for sdpa_ocl non-dynamic with prefill_stage \n";
                    add_stage(ocl_multi_tokens, params);
                    // add_stage() swallows a codegen failure; fall back to the opt kernel then.
                    if (!has_stage(ocl_multi_tokens)) {
                        GPU_DEBUG_TRACE_DETAIL << "fail to create sdpa_ocl kernel, fallback to regular_multi_tokens for prefill \n";
                        add_stage(regular_multi_tokens, params);
                    }
#endif
                } else {
                    GPU_DEBUG_TRACE_DETAIL << "add stage regular_multi_tokens kernels with prefill_stage \n";
                    add_stage(regular_multi_tokens, params);
                }
            } else {
                GPU_DEBUG_TRACE_DETAIL << "add stage single_tokens \n";
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
                const bool use_sdpa_ocl = ocl_single_token_ok && dpas_supported() &&
                                          (cldnn::paged_attention::sdpa_ocl_plain_single_token_ok(params.get_device_info()) || needs_ocl_single_token) &&
                                          !is_indirect && single_query_row(params);
                if (use_sdpa_ocl) {
                    add_stage(ocl_single_token, params);
                }
#endif
                if (unaligned_kv_head(params) && !has_stage(ocl_single_token)) {
                    // Unaligned decode on the sdpa_ocl lane whose sdpa_ocl stage was refused or could not be added:
                    // execute() runs regular_multi_tokens, the only opt kernel that handles unaligned heads.
                    GPU_DEBUG_TRACE_DETAIL << "no sdpa_ocl single-token stage for unaligned head size, add regular_multi_tokens \n";
                    add_stage(regular_multi_tokens, params);
                } else {
                    add_stage(is_indirect ? indirect_single_token : regular_single_token, params);
                    if (get_partitions_num(params, SDPAStage::SINGLE_TOKEN) > 1) {
                        add_stage(is_indirect ? indirect_finalization : regular_finalization, params);
                    }
                }
            }
        }
    }

    [[nodiscard]] event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        const auto& params = *instance.get_impl_params();
        auto new_params = SDPABase::requires_shape_canonicalization(params) ? SDPABase::static_canonicalize_shapes(params) : params;
        const bool multi_token_rows = use_multi_token_rows(new_params);
        bool is_indirect = need_indirect_load(static_cast<scaled_dot_product_attention_inst&>(instance));
        GPU_DEBUG_TRACE_DETAIL << "execute indirect = " << is_indirect << ", multi_token_rows = " << multi_token_rows << "\n";
        update_rt_params(instance);

        kernel_dump_info.clear_entries();

#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
        const auto k_dt = new_params.input_layouts[1].data_type;
        const bool is_kv_compressed = data_type_traits::is_i8_u8(k_dt) || data_type_traits::is_i4_u4(k_dt);
        const auto kv_cache_dt = new_params.get_program().get_config().get_kv_cache_precision();
        const bool is_int4_kv = is_kv_compressed && ov::element::Type(kv_cache_dt).bitwidth() == 4;

        // Per-channel compressed KV is read by sdpa_ocl (validate_impl() admits it where sdpa_ocl serves it) and on xe_hpg by
        // the opt single-token stage (KV_COMP_PER_CHANNEL), which the ctor leaves its one-query steps to where it takes the
        // head; every other opt stage reads one scale per token, so a missing stage must fail here rather than give wrong
        // output. A plain int4 cache otherwise stays on the opt kernels (sdpa_ocl unpacks the nibbles of the per-channel one only).
        const bool per_channel_kv = SDPABase::has_per_channel_compressed_kv(new_params);
        if (per_channel_kv) {
            const bool opt_decode =
                !multi_token_rows && !has_stage(ocl_single_token) && SDPABase::opt_single_token_reads_per_channel(new_params) && !unaligned_kv_head(new_params);
            if (!opt_decode) {
                OPENVINO_ASSERT(!is_indirect && has_stage(multi_token_rows ? ocl_multi_tokens : ocl_single_token),
                                "[GPU] SDPA with per-channel compressed KV needs the sdpa_ocl ",
                                multi_token_rows ? "multi-token" : "single-token",
                                " stage (indirect = ",
                                is_indirect,
                                ")");
                return execute_stage(events, instance, multi_token_rows ? ocl_multi_tokens : ocl_single_token);
            }
            OPENVINO_ASSERT(!is_indirect, "[GPU] SDPA with per-channel compressed KV has no beam-table reader");
        }
        // A transposed V likewise (validate_impl() admits it where sdpa_ocl serves it): the opt stages read V's head dim as
        // contiguous, but for the single-token stage on xe_hpg (V_TRANSPOSED_READ), which the ctor leaves its one-query steps
        // to where it takes the head; those go on to the opt single-token path below.
        if (SDPABase::has_transposed_v(new_params)) {
            const bool opt_decode = !multi_token_rows && !has_stage(ocl_single_token) && SDPABase::opt_single_token_reads_transposed_v(new_params) &&
                                    !unaligned_kv_head(new_params);
            if (!opt_decode) {
                OPENVINO_ASSERT(!is_indirect && has_stage(multi_token_rows ? ocl_multi_tokens : ocl_single_token),
                                "[GPU] SDPA with a transposed V needs the sdpa_ocl ",
                                multi_token_rows ? "multi-token" : "single-token",
                                " stage (indirect = ",
                                is_indirect,
                                ")");
                return execute_stage(events, instance, multi_token_rows ? ocl_multi_tokens : ocl_single_token);
            }
            OPENVINO_ASSERT(!is_indirect, "[GPU] SDPA with a transposed V has no beam-table reader");
        }
        if (multi_token_rows && !is_indirect && !is_int4_kv && has_stage(ocl_multi_tokens)) {
            GPU_DEBUG_TRACE_DETAIL << "execute ocl_multi_tokens for prefill \n";
            return execute_stage(events, instance, ocl_multi_tokens);
        }
#endif
        // TODO: Unaligned head size is currently supported by only multi tokens kernel.
        // So far this case was observed only from the non-lm models such as vision embedding model.
        // If we need to optimize unaligned head size SDPA for 2nd+ token phase of LM model,
        // we'll need to fix single_token kernel to support unaligned head size.
        // The sdpa_ocl single-token kernel does support unaligned head sizes (head < d / value < dv /
        // DKS_ACTIVE guards). When it is staged (sdpa_ocl lane, non-indirect decode) it is used instead;
        // when it is not, the ctor stages regular_multi_tokens for a static non-indirect unaligned decode.
        if (multi_token_rows || (unaligned_kv_head(new_params) && (is_indirect || !has_stage(ocl_single_token)))) {
            GPU_DEBUG_TRACE_DETAIL << "execute multi_tokens for prefill with indirect = " << is_indirect << "\n";
            return execute_stage(events, instance, is_indirect ? indirect_multi_tokens : regular_multi_tokens);
        }
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
        if (!is_indirect && !is_int4_kv && single_query_row(new_params) && has_stage(ocl_single_token)) {
            return execute_stage(events, instance, ocl_single_token);
        }
#endif
        const auto num_of_partitions = get_partitions_num(new_params, SDPAStage::SINGLE_TOKEN);
        GPU_DEBUG_TRACE_DETAIL << "execute single_tokens with indirect = " << is_indirect << "\n";
        auto ev = execute_stage(events, instance, is_indirect ? indirect_single_token : regular_single_token);
        if (num_of_partitions > 1) {
            GPU_DEBUG_TRACE_DETAIL << "execute single_tokens_finalization (" << num_of_partitions << ") with indirect = " << is_indirect << "\n";
            ev = execute_stage({ev}, instance, is_indirect ? indirect_finalization : regular_finalization);
        }
        return ev;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        std::vector<BufferDescriptor> internal_buffers;
        auto params_canonicalization = SDPABase::requires_shape_canonicalization(params) ? SDPABase::static_canonicalize_shapes(params) : params;
        // exp_sums and max_logits hold one value per output row and partition; the output is V_HEAD_SIZE wide
        const auto v_head_size = SDPABase::get_effective_v_head_size(params_canonicalization);

        const auto num_of_partitions = get_partitions_num(params_canonicalization, SDPAStage::SINGLE_TOKEN);
        const auto multi_token_rows = use_multi_token_stage(params_canonicalization);
        const size_t buf_elements_count = (num_of_partitions == 1 || multi_token_rows) ? 1 : params.output_layouts[0].count() / v_head_size * num_of_partitions;
        const size_t tmp_out_elements_count = (num_of_partitions == 1 || multi_token_rows) ? 2 : params.output_layouts[0].count() * num_of_partitions;

        internal_buffers.emplace_back(buf_elements_count, ov::element::f32);
        internal_buffers.emplace_back(buf_elements_count, ov::element::f32);
        internal_buffers.emplace_back(tmp_out_elements_count, params.output_layouts[0].data_type);

        return internal_buffers;
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SDPAOptImpl>(this);
    }
};

bool SDPAOpt::has_bf16_compressed_kv_zero_point(const kernel_impl_params& params) {
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
    const auto desc = params.typed_desc<scaled_dot_product_attention>();
    if (!desc->is_kv_compressed || desc->quantization_attributes.quantization_type != ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric ||
        desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar) {
        return false;
    }

    const auto& k_layout = params.get_input_layout(1);
    if (!data_type_traits::is_i8_u8(k_layout.data_type) && !data_type_traits::is_i4_u4(k_layout.data_type)) {
        return false;
    }

    // Per-channel compressed KV runs on the sdpa_ocl stages only (validate_impl(), execute()), never on the regular opt ones.
    if (has_per_channel_compressed_kv(params)) {
        return false;
    }

    const auto data_inputs_num = get_data_inputs_num(*desc);
    return params.get_input_layout(data_inputs_num + 2).data_type == ov::element::bf16;
#else
    UNUSED(params);
    return false;
#endif
}

bool SDPAOpt::dpas_lane_op_supported(const RuntimeParams& params, bool allow_scalar_mask) {
#ifdef OV_GPU_WITH_ONEDNN_MICROKERNELS
    // dpas_sdpa_op_supported() reads 4D layouts; a rank-3 op reaches here with rank-3 ones.
    const SDPACanonicalParams canonical(params);
    return cldnn::paged_attention::sdpa_ocl_selected(params.get_device_info()) && dpas_sdpa_enabled(params) &&
           dpas_sdpa_op_supported(canonical.get(), allow_scalar_mask);
#else
    UNUSED(params);
    UNUSED(allow_scalar_mask);
    return false;
#endif
}

bool SDPAOpt::sdpa_ocl_serves_op(const RuntimeParams& params, bool allow_scalar_mask) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    return dpas_lane_op_supported(params, allow_scalar_mask) && SDPAOclGenerator::supported(params);
#else
    UNUSED(params);
    UNUSED(allow_scalar_mask);
    return false;
#endif
}

bool SDPAOpt::dpas_sdpa_op_supported(const RuntimeParams& params, bool allow_scalar_mask) {
#ifdef ENABLE_ONEDNN_FOR_GPU
    auto desc = params.typed_desc<scaled_dot_product_attention>();

    const auto& q_layout = params.get_input_layout(0);
    const auto& k_layout = params.get_input_layout(1);
    const auto& v_layout = params.get_input_layout(2);

    auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
    auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
    auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);

    ov::Dimension Q_num_heads_dim = get_num_heads(q_layout, extended_input_q_transpose_order);
    ov::Dimension K_num_heads_dim = get_num_heads(k_layout, extended_input_k_transpose_order);
    ov::Dimension V_num_heads_dim = get_num_heads(v_layout, extended_input_v_transpose_order);

    // The head dim must stay innermost, except for a transposed V (order {..,3,2}) that sdpa_ocl's v_tile_t reads
    // (transposed_v_reader_ok()); every other V read assumes a unit head-dim stride.
    if (extended_input_q_transpose_order[3] != 3 || extended_input_k_transpose_order[3] != 3 ||
        (extended_input_v_transpose_order[3] != 3 && !SDPABase::transposed_v_reader_ok(params))) {
        return false;
    }

    if (Q_num_heads_dim.is_dynamic() || K_num_heads_dim.is_dynamic() || V_num_heads_dim.is_dynamic() || K_num_heads_dim != V_num_heads_dim) {
        return false;
    }

    auto K_head_size = get_head_size(k_layout, extended_input_k_transpose_order);
    auto V_head_size = get_head_size(v_layout, extended_input_v_transpose_order);
    if (K_head_size != V_head_size || K_head_size > 512 || V_head_size > 512) {
        return false;
    }

    auto data_inputs_num = get_data_inputs_num(*desc);
    // A single-element runtime attention mask (rank-0 scalar or 1-element 1D placeholder): the sdpa_ocl kernel reads
    // the one value and adds it to every logit. Rejected unless the caller allows it (validate_impl() does not).
    const auto mask_idx = 3LU;
    if (!desc->attn_mask_val.has_value() && data_inputs_num > mask_idx && !params.get_input_layout(mask_idx).is_dynamic() &&
        params.get_input_layout(mask_idx).count() == 1 && !allow_scalar_mask) {
        return false;
    }
    if (q_layout.get_partial_shape()[mask_idx].get_length() > 512) {
        return false;
    }

    // A rank-3 op (Q/K/V [B, L, E]) takes the DPAS kernel only where sdpa_ocl_plain_3d_ok() / _head_ok() accept it; everywhere
    // else it runs on sdpa_opt ("Rank-3 plain SDPA" in the docs).
    const bool is_output_rank_4d = desc->output_transpose_order.size() == 4;

    // Asymmetric compressed KV with a bf16 zero-point buffer stays off the DPAS kernels (the removed micro stage gave NaN on it).
    if (has_bf16_compressed_kv_zero_point(params)) {
        return false;
    }

    return is_output_rank_4d ||
           (cldnn::paged_attention::sdpa_ocl_plain_3d_ok(params.get_device_info()) && cldnn::paged_attention::sdpa_ocl_plain_3d_head_ok(K_head_size));
#else
    UNUSED(params);
    UNUSED(allow_scalar_mask);
    return false;
#endif
}

std::unique_ptr<primitive_impl> SDPAOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    try {
        return std::make_unique<SDPAOptImpl>(params);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Failed to create SDPAOptImpl: ", e.what());
    }
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::scaled_dot_product_attention)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SDPAOptImpl)
