// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_gemm_onednn.hpp"
#include "moe_gemm_otd_context.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"

#include <oneapi/dnnl/dnnl.hpp>
#ifdef OV_GPU_WITH_ZE_RT
#include <oneapi/dnnl/dnnl_ze.hpp>
#else
#include <oneapi/dnnl/dnnl_ocl.hpp>
#endif

#include <algorithm>
#include <memory>
#include <map>
#include <utility>

namespace cldnn {
namespace onednn {

struct moe_gemm_onednn : typed_primitive_onednn_impl<moe_gemm> {
    using parent = typed_primitive_onednn_impl<moe_gemm>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::moe_gemm_onednn)

    // OTD members
    mutable std::shared_ptr<Moe2GemmOtdContext> _otd_ctx;
    mutable memory::ptr _scratch_offsets;  // [lru_expert_num] i32 scratch for remapped offsets
    mutable memory::ptr _scratch_bias;     // [lru_expert_num, N] scratch for per-batch gathered bias
    bool _is_up_gemm = true;
    size_t _lru_expert_num = 0;

    // Per-batch primitives for prefill overflow: built lazily, keyed by (M, num_groups).
    // Each batch of the overflow split runs a grouped matmul with M = batch token count.
    mutable std::shared_ptr<dnnl::primitive_attr> _otd_attr;
    struct BatchPrim {
        dnnl::matmul::primitive_desc pd;
        dnnl::matmul prim;
        dnnl::memory::desc scratchpad_md;  // scratchpad requirement for this PD (mode=user)
        memory::ptr scratchpad;            // backing USM allocation for the scratchpad
    };
    mutable std::map<std::pair<int64_t, int64_t>, BatchPrim> _batch_prim_cache;
    // Cached geometry for building per-batch PDs at runtime
    dnnl::memory::dim _otd_N = 0;
    dnnl::memory::dim _otd_K = 0;
    dnnl::memory::data_type _otd_src_dt = dnnl::memory::data_type::undef;
    dnnl::memory::data_type _otd_dst_dt = dnnl::memory::data_type::undef;
    dnnl::memory::data_type _otd_wei_dt = dnnl::memory::data_type::undef;
    dnnl::memory::data_type _otd_bias_dt = dnnl::memory::data_type::undef;
    bool _otd_has_bias = false;

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<moe_gemm_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(moe_gemm_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args;
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());

        {
            auto& input = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT);
            auto& offsets = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
            dnnl::memory input_mem = input.get_onednn_grouped_memory(_pd.src_desc(0), offsets);
            args.insert({DNNL_ARG_SRC, input_mem});
        }

        {
            auto& output = instance.output_memory(0);
            auto& offsets = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
            dnnl::memory output_mem = output.get_onednn_grouped_memory(_pd.dst_desc(0), offsets);
            args.insert({DNNL_ARG_DST, output_mem});
        }

        {
            auto& weights = instance.input_memory(moe_gemm::MoEGemmInputIdx::WEIGHT);
            dnnl::memory weights_mem = weights.get_onednn_memory(_pd.weights_desc(0), 0);
            args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        }

        if (moe_cfg.is_weight_quantized) {
            // cldnn [E,N,G] -> onednn [E,G,N] (matches byfx physical from prepare_quantization).
            auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
            auto wei_scales_shape = wei_scales.get_layout().get_shape();
            dnnl::memory::dim d0 = wei_scales_shape[0];
            dnnl::memory::dim d1 = wei_scales_shape[1];
            dnnl::memory::dim d2 = wei_scales_shape[2];
            // Cross-check moe_cfg.weight_group_size (from compile-time scale_shape[2]) vs runtime memory.
            const auto& weight_layout = instance.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
            const auto& w_shape = weight_layout.get_shape();
            const dnnl::memory::dim K = (w_shape.size() == 4) ? w_shape[2] * w_shape[3] : w_shape[2];
            const dnnl::memory::dim runtime_num_groups = (moe_cfg.weight_group_size == -1)
                ? 1
                : (K / moe_cfg.weight_group_size);
            const dnnl::memory::dim scale_num_groups = (wei_scales_shape.size() >= 3) ? d2 : 1;
            OPENVINO_ASSERT(scale_num_groups == runtime_num_groups,
                            "moe_gemm scale shape ", wei_scales_shape, " implies num_groups=",
                            scale_num_groups, " but moe_cfg.weight_group_size=", moe_cfg.weight_group_size,
                            " (K=", K, ") implies ", runtime_num_groups);
            dnnl::memory::dims wei_scales_dims = (moe_cfg.weight_group_size == -1)
                ? dnnl::memory::dims{d0, d1}
                : dnnl::memory::dims{d0, d2, d1};
            dnnl::memory::format_tag wei_scales_fmt = (moe_cfg.weight_group_size == -1)
                ? dnnl::memory::format_tag::ab
                : dnnl::memory::format_tag::abc;
            dnnl::memory::desc wei_scales_md(
                wei_scales_dims, convert_data_type(wei_scales.get_layout().data_type), wei_scales_fmt);
            dnnl::memory wei_scales_mem = wei_scales.get_onednn_memory(wei_scales_md, 0);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem});

            if (!moe_cfg.is_weight_symmetric_quantized) {
                auto& wei_zp = instance.input_memory(moe_cfg.weight_zp_idx);
                const auto& zp_shape = wei_zp.get_layout().get_shape();
                OPENVINO_ASSERT(zp_shape == wei_scales_shape,
                                "moe_gemm scale shape ", wei_scales_shape, " does not match zp shape ", zp_shape);
                dnnl::memory::desc wei_zp_md(
                    wei_scales_dims, convert_data_type(wei_zp.get_layout().data_type), wei_scales_fmt);
                dnnl::memory wei_zp_mem = wei_zp.get_onednn_memory(wei_zp_md, 0);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp_mem});
            }
        }

        if (moe_cfg.has_bias) {
            auto& bias = instance.input_memory(moe_gemm::MoEGemmInputIdx::BIAS);
            dnnl::memory bias_mem = bias.get_onednn_memory(_pd.weights_desc(1), 0);
            args.insert({DNNL_ARG_BIAS, bias_mem});
        }

        return args;
    }

    // OTD-aware get_arguments: uses remapped offset buffer for input/output grouping
    std::unordered_map<int, dnnl::memory> get_arguments_otd(moe_gemm_inst& instance) const {
        std::unordered_map<int, dnnl::memory> args;
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());

        // Input/Output use the SCRATCH offset buffer (slot-remapped)
        {
            auto& input = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT);
            dnnl::memory input_mem = input.get_onednn_grouped_memory(_pd.src_desc(0), *_scratch_offsets);
            args.insert({DNNL_ARG_SRC, input_mem});
        }
        {
            auto& output = instance.output_memory(0);
            dnnl::memory output_mem = output.get_onednn_grouped_memory(_pd.dst_desc(0), *_scratch_offsets);
            args.insert({DNNL_ARG_DST, output_mem});
        }
        // Weight/scales/zp from resident buffers (sized for lru_expert_num)
        {
            auto& weights = instance.input_memory(moe_gemm::MoEGemmInputIdx::WEIGHT);
            dnnl::memory weights_mem = weights.get_onednn_memory(_pd.weights_desc(0), 0);
            args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        }
        if (moe_cfg.is_weight_quantized) {
            auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
            auto wei_scales_shape = wei_scales.get_layout().get_shape();
            dnnl::memory::dim d0 = static_cast<dnnl::memory::dim>(_lru_expert_num);
            dnnl::memory::dim d1 = wei_scales_shape[1];
            dnnl::memory::dim d2 = wei_scales_shape[2];
            dnnl::memory::dims wei_scales_dims = (moe_cfg.weight_group_size == -1)
                ? dnnl::memory::dims{d0, d1}
                : dnnl::memory::dims{d0, d2, d1};
            dnnl::memory::format_tag wei_scales_fmt = (moe_cfg.weight_group_size == -1)
                ? dnnl::memory::format_tag::ab
                : dnnl::memory::format_tag::abc;
            dnnl::memory::desc wei_scales_md(
                wei_scales_dims, convert_data_type(wei_scales.get_layout().data_type), wei_scales_fmt);
            dnnl::memory wei_scales_mem = wei_scales.get_onednn_memory(wei_scales_md, 0);
            args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem});

            if (!moe_cfg.is_weight_symmetric_quantized) {
                auto& wei_zp = instance.input_memory(moe_cfg.weight_zp_idx);
                dnnl::memory::desc wei_zp_md(
                    wei_scales_dims, convert_data_type(wei_zp.get_layout().data_type), wei_scales_fmt);
                dnnl::memory wei_zp_mem = wei_zp.get_onednn_memory(wei_zp_md, 0);
                args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp_mem});
            }
        }
        if (moe_cfg.has_bias) {
            auto& bias = instance.input_memory(moe_gemm::MoEGemmInputIdx::BIAS);
            dnnl::memory bias_mem = bias.get_onednn_memory(_pd.weights_desc(1), 0);
            args.insert({DNNL_ARG_BIAS, bias_mem});
        }
        return args;
    }

    // OTD pre-execution: load needed experts from disk into the weight buffer
    // Returns true if execution was handled internally (batched), false if caller should execute.
    bool prepare_otd_execution(moe_gemm_inst& instance) const {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();

        // Bind resident buffers — each direction binds its own buffers independently
        if (_is_up_gemm && !_otd_ctx->resident.up_w) {
            auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());
            _otd_ctx->resident.up_w = instance.input_memory_ptr(moe_gemm::MoEGemmInputIdx::WEIGHT);
            if (moe_cfg.is_weight_quantized)
                _otd_ctx->resident.up_s = instance.input_memory_ptr(moe_cfg.weight_scale_idx);
            if (moe_cfg.is_weight_quantized && !moe_cfg.is_weight_symmetric_quantized)
                _otd_ctx->resident.up_z = instance.input_memory_ptr(moe_cfg.weight_zp_idx);
            _otd_ctx->bound = true;
        } else if (!_is_up_gemm && !_otd_ctx->resident.down_w) {
            auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());
            _otd_ctx->resident.down_w = instance.input_memory_ptr(moe_gemm::MoEGemmInputIdx::WEIGHT);
            if (moe_cfg.is_weight_quantized)
                _otd_ctx->resident.down_s = instance.input_memory_ptr(moe_cfg.weight_scale_idx);
            if (moe_cfg.is_weight_quantized && !moe_cfg.is_weight_symmetric_quantized)
                _otd_ctx->resident.down_z = instance.input_memory_ptr(moe_cfg.weight_zp_idx);
            _otd_ctx->bound = true;
        }

        // Read expert IDs from GPU
        auto& experts_ids_mem = instance.input_memory(moe_gemm::MoEGemmInputIdx::EXPERTS_IDS);
        const auto& experts_shape = experts_ids_mem.get_layout().get_shape();
        const size_t num_active = experts_shape[0];

        std::vector<int32_t> expert_ids_cpu(num_active);
        experts_ids_mem.copy_to(stream, expert_ids_cpu.data(), 0, 0, num_active * sizeof(int32_t), true);

        // Deduplicate and sort active expert IDs
        std::vector<uint32_t> unique_experts;
        unique_experts.reserve(num_active);
        for (size_t i = 0; i < num_active; i++) {
            unique_experts.push_back(static_cast<uint32_t>(expert_ids_cpu[i]));
        }
        std::sort(unique_experts.begin(), unique_experts.end());
        unique_experts.erase(std::unique(unique_experts.begin(), unique_experts.end()), unique_experts.end());

        // Read original offset buffer from GPU (needed for both overflow and normal paths)
        auto& orig_offsets_mem = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT_OFFSET_PER_EXPERT);
        const auto& orig_shape = orig_offsets_mem.get_layout().get_shape();
        const size_t num_experts_in_offset = orig_shape[0];

        std::vector<int32_t> orig_offsets(num_experts_in_offset);
        orig_offsets_mem.copy_to(stream, orig_offsets.data(), 0, 0, num_experts_in_offset * sizeof(int32_t), true);

        // Always use the per-batch execution path. When unique_experts <= capacity,
        // this is a single batch (no pointer shift). When it exceeds capacity, it splits
        // into chunks. Each batch builds a PD with M = batch token count so the grouped
        // matmul constraint (last_offset == M) is satisfied.
        //
        // For decode (few tokens), take the swap-free per-expert path: each expert's weight
        // is read directly from its resident LRU slot, avoiding the GPU->GPU relocations that
        // the contiguous-slot grouped path needs. Detected by total gathered rows being small
        // (<= capacity), which guarantees all active experts fit resident simultaneously.
        const int32_t total_tokens = unique_experts.empty()
            ? 0
            : orig_offsets[unique_experts.back()];
        const bool is_decode = total_tokens > 0 &&
                               total_tokens <= static_cast<int32_t>(_lru_expert_num);
        if (is_decode) {
            execute_decode(instance, stream, unique_experts, orig_offsets);
        } else {
            execute_batched(instance, stream, unique_experts, orig_offsets);
        }
        return true;  // Execution already done
    }

    // Helper to create a grouped dnnl::memory with a byte offset on the data pointer.
    // This allows the grouped matmul to operate on a shifted region of the SRC/DST buffer.
    // offsets_byte_offset shifts the group-offset buffer pointer too, so several single-group
    // calls can share one offsets scratch buffer (each reads its own entry).
    dnnl::memory make_shifted_grouped_memory(const dnnl::memory::desc& desc,
                                              const memory& data_mem,
                                              const memory& offsets_mem,
                                              int64_t data_byte_offset,
                                              const engine& eng,
                                              int64_t offsets_byte_offset = 0) const {
        void* data_ptr = reinterpret_cast<uint8_t*>(data_mem.buffer_ptr()) + data_byte_offset;
        void* offsets_ptr = reinterpret_cast<uint8_t*>(offsets_mem.buffer_ptr()) + offsets_byte_offset;
        auto onednn_engine = eng.get_onednn_engine();
#ifdef OV_GPU_WITH_ZE_RT
        return dnnl::ze_interop::make_memory(desc, onednn_engine, std::vector<void*>{data_ptr, offsets_ptr});
#else
        return dnnl::ocl_interop::make_memory(desc, onednn_engine, dnnl::ocl_interop::memory_kind::usm,
            std::vector<void*>{data_ptr, offsets_ptr});
#endif
    }

    // Build (or fetch from cache) a grouped matmul primitive for a specific
    // (M = batch token count, num_groups = experts in this batch).
    // Used for prefill overflow where the number of active experts exceeds capacity.
    BatchPrim&
    get_batch_primitive(int64_t M, int64_t num_groups, engine& eng) const {
        auto key = std::make_pair(M, num_groups);
        auto it = _batch_prim_cache.find(key);
        if (it != _batch_prim_cache.end())
            return it->second;

        auto onednn_engine = eng.get_onednn_engine();
        dnnl::memory::dims input_dims = {M, _otd_K};
        dnnl::memory::dims weights_dims = {num_groups, _otd_K, _otd_N};
        dnnl::memory::dims output_dims = {M, _otd_N};

        auto input_md = dnnl::memory::desc::grouped(input_dims, _otd_src_dt, 0, num_groups);
        auto output_md = dnnl::memory::desc::grouped(output_dims, _otd_dst_dt, 0, num_groups);
        auto weights_md = dnnl::memory::desc(weights_dims, _otd_wei_dt, dnnl::memory::format_tag::acb);

        dnnl::matmul::primitive_desc pd = _otd_has_bias
            ? dnnl::matmul::primitive_desc(onednn_engine, input_md, weights_md,
                  dnnl::memory::desc({num_groups, _otd_N}, _otd_bias_dt, dnnl::memory::format_tag::ab),
                  output_md, *_otd_attr)
            : dnnl::matmul::primitive_desc(onednn_engine, input_md, weights_md, output_md, *_otd_attr);
        dnnl::matmul prim(pd);

        // Scratchpad mode is 'user' (see program_node.cpp), so the caller must supply
        // DNNL_ARG_SCRATCHPAD. Allocate a backing buffer sized for this PD's requirement.
        auto scratchpad_md = pd.scratchpad_desc();
        memory::ptr scratchpad;
        if (scratchpad_md.get_size() > 0) {
            auto sp_layout = cldnn::layout(
                ov::Shape{scratchpad_md.get_size()}, cldnn::data_types::u8, cldnn::format::bfyx);
            scratchpad = eng.allocate_memory(sp_layout, cldnn::allocation_type::usm_device, false);
        }

        BatchPrim bp{std::move(pd), std::move(prim), std::move(scratchpad_md), std::move(scratchpad)};
        auto res = _batch_prim_cache.emplace(key, std::move(bp));
        return res.first->second;
    }


    // Decode-optimized path (small token count). Instead of forcing experts into contiguous
    // slots for a single grouped matmul (which requires GPU->GPU swaps every token), acquire
    // each expert's resident LRU slot and issue one single-group matmul per expert, reading
    // its weight/scale/zp directly from that slot via a byte offset. Resident experts stay in
    // place (zero copy on a cache hit), mirroring the 3GEMM gather-matmul behaviour.
    void execute_decode(moe_gemm_inst& instance, stream& stream,
                        const std::vector<uint32_t>& all_experts,
                        const std::vector<int32_t>& orig_offsets) const {
        auto& network = instance.get_network();
        auto& eng = network.get_engine();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());

        auto& input = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT);
        auto& output = instance.output_memory(0);
        auto& weights = instance.input_memory(moe_gemm::MoEGemmInputIdx::WEIGHT);

        auto src_layout = input.get_layout();
        auto dst_layout = output.get_layout();
        const size_t src_row_bytes = src_layout.get_shape().back() * ov::element::Type(src_layout.data_type).size();
        const size_t dst_row_bytes = dst_layout.get_shape().back() * ov::element::Type(dst_layout.data_type).size();

        const size_t num_experts_total = _otd_ctx->num_experts;
        const size_t weight_per_expert = weights.get_layout().bytes_count() / num_experts_total;

        // Per-expert scale/zp geometry (resident buffers reinterpreted to logical num_experts).
        size_t scale_per_expert = 0;
        size_t zp_per_expert = 0;
        dnnl::memory::dim sc_oc = 0;  // scale/zp OC dim (shape[1])
        dnnl::memory::dim sc_g = 0;   // scale/zp group dim (shape[2])
        if (moe_cfg.is_weight_quantized) {
            auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
            scale_per_expert = wei_scales.get_layout().bytes_count() / num_experts_total;
            const auto ss = wei_scales.get_layout().get_shape();
            sc_oc = static_cast<dnnl::memory::dim>(ss[1]);
            sc_g = static_cast<dnnl::memory::dim>(ss[2]);
            if (!moe_cfg.is_weight_symmetric_quantized) {
                auto& wei_zp = instance.input_memory(moe_cfg.weight_zp_idx);
                zp_per_expert = wei_zp.get_layout().bytes_count() / num_experts_total;
            }
        }

        // Pure-LRU acquire: each expert -> resident slot (hit = no copy, miss = disk load).
        auto slots = _otd_ctx->acquire_experts_lru(stream, all_experts, _is_up_gemm);
        stream.finish();  // ensure any disk uploads complete before the matmuls read them

        const int64_t n = static_cast<int64_t>(all_experts.size());
        for (int64_t i = 0; i < n; i++) {
            const uint32_t expert = all_experts[i];
            const int32_t token_start = (i == 0) ? 0 : orig_offsets[all_experts[i - 1]];
            const int64_t M = static_cast<int64_t>(orig_offsets[expert]) - token_start;
            if (M <= 0)
                continue;
            const size_t slot = slots[i];

            // Write this expert's single group-offset (= M) at index i so concurrent enqueues
            // do not clobber each other's offset entry before execution.
            int32_t off_val = static_cast<int32_t>(M);
            _scratch_offsets->copy_from(stream, &off_val, 0, static_cast<size_t>(i) * sizeof(int32_t),
                                        sizeof(int32_t), true);

            auto& pd_prim = get_batch_primitive(M, 1, eng);
            auto& pd = pd_prim.pd;

            std::unordered_map<int, dnnl::memory> args;
            const int64_t src_off = static_cast<int64_t>(token_start) * static_cast<int64_t>(src_row_bytes);
            const int64_t dst_off = static_cast<int64_t>(token_start) * static_cast<int64_t>(dst_row_bytes);
            const int64_t off_bytes = i * static_cast<int64_t>(sizeof(int32_t));
            args.insert({DNNL_ARG_SRC, make_shifted_grouped_memory(pd.src_desc(), input, *_scratch_offsets, src_off, eng, off_bytes)});
            args.insert({DNNL_ARG_DST, make_shifted_grouped_memory(pd.dst_desc(), output, *_scratch_offsets, dst_off, eng, off_bytes)});
            args.insert({DNNL_ARG_WEIGHTS, weights.get_onednn_memory(pd.weights_desc(),
                            static_cast<int64_t>(slot * weight_per_expert))});

            if (pd_prim.scratchpad)
                args.insert({DNNL_ARG_SCRATCHPAD, pd_prim.scratchpad->get_onednn_memory(pd_prim.scratchpad_md, 0)});

            if (moe_cfg.is_weight_quantized) {
                auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
                dnnl::memory::dims sdims = (moe_cfg.weight_group_size == -1)
                    ? dnnl::memory::dims{1, sc_oc}
                    : dnnl::memory::dims{1, sc_g, sc_oc};
                dnnl::memory::format_tag sfmt = (moe_cfg.weight_group_size == -1)
                    ? dnnl::memory::format_tag::ab
                    : dnnl::memory::format_tag::abc;
                dnnl::memory::desc smd(sdims, convert_data_type(wei_scales.get_layout().data_type), sfmt);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                             wei_scales.get_onednn_memory(smd, static_cast<int64_t>(slot * scale_per_expert))});

                if (!moe_cfg.is_weight_symmetric_quantized) {
                    auto& wei_zp = instance.input_memory(moe_cfg.weight_zp_idx);
                    dnnl::memory::desc zmd(sdims, convert_data_type(wei_zp.get_layout().data_type), sfmt);
                    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
                                 wei_zp.get_onednn_memory(zmd, static_cast<int64_t>(slot * zp_per_expert))});
                }
            }

            if (_otd_has_bias) {
                auto& bias = instance.input_memory(moe_gemm::MoEGemmInputIdx::BIAS);
                // Bias is a full [num_experts, N] buffer (not OTD-compacted): index by expert id.
                const size_t bias_row_bytes =
                    static_cast<size_t>(_otd_N) * dnnl::memory::data_type_size(_otd_bias_dt);
                dnnl::memory::desc bmd({1, _otd_N}, _otd_bias_dt, dnnl::memory::format_tag::ab);
                args.insert({DNNL_ARG_BIAS, bias.get_onednn_memory(bmd,
                                static_cast<int64_t>(expert * bias_row_bytes))});
            }

            try {
                pd_prim.prim.execute(stream.get_onednn_stream(), args);
            } catch (dnnl::error& err) {
                OPENVINO_THROW("OTD decode execution failed at expert=", expert,
                    " M=", M, " slot=", slot, " error: ", err.what());
            }
        }
        stream.finish();
    }


    void execute_batched(moe_gemm_inst& instance, stream& stream,
                         const std::vector<uint32_t>& all_experts,
                         const std::vector<int32_t>& orig_offsets) const {
        auto& network = instance.get_network();
        auto& eng = network.get_engine();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(*instance.get_impl_params());

        auto& input = instance.input_memory(moe_gemm::MoEGemmInputIdx::INPUT);
        auto& output = instance.output_memory(0);
        auto& weights = instance.input_memory(moe_gemm::MoEGemmInputIdx::WEIGHT);

        // Compute element strides for SRC and DST
        auto src_layout = input.get_layout();
        auto dst_layout = output.get_layout();
        size_t src_row_bytes = src_layout.get_shape().back() * ov::element::Type(src_layout.data_type).size();
        size_t dst_row_bytes = dst_layout.get_shape().back() * ov::element::Type(dst_layout.data_type).size();

        const size_t total = all_experts.size();
        for (size_t batch_start = 0; batch_start < total; batch_start += _lru_expert_num) {
            size_t batch_end = std::min(batch_start + _lru_expert_num, total);
            std::vector<uint32_t> batch_experts(all_experts.begin() + batch_start,
                                                 all_experts.begin() + batch_end);
            const int64_t num_groups = static_cast<int64_t>(batch_experts.size());

            // Load this batch of experts into slots
            _otd_ctx->load_experts(stream, batch_experts, _is_up_gemm);
            // Ensure the weight/scale/zp uploads complete before the matmul reads them.
            stream.finish();

            // Compute the token start position for this batch:
            // batch_token_start = end offset of the expert just before this batch.
            int32_t batch_token_start = 0;
            if (batch_start > 0) {
                uint32_t prev_expert = all_experts[batch_start - 1];
                batch_token_start = orig_offsets[prev_expert];
            }

            // Build relative offset buffer for this batch. Because the grouped matmul
            // requires the last offset to equal M, and M for this batch is the batch's
            // token count, we make offsets relative to batch_token_start.
            std::vector<int32_t> remapped_offsets(num_groups, 0);
            int32_t last_offset = 0;
            for (int64_t i = 0; i < num_groups; i++) {
                last_offset = orig_offsets[batch_experts[i]] - batch_token_start;
                remapped_offsets[i] = last_offset;
            }
            const int64_t M = last_offset;  // total tokens in this batch

            if (M == 0)
                continue;  // no tokens for this batch (shouldn't happen for active experts)

            _scratch_offsets->copy_from(stream, remapped_offsets.data(), 0, 0,
                                        num_groups * sizeof(int32_t), true);

            // Build (or fetch) a per-batch primitive with M and num_groups
            auto& pd_prim = get_batch_primitive(M, num_groups, eng);
            auto& batch_pd = pd_prim.pd;
            auto& batch_prim = pd_prim.prim;

            // Build arguments with shifted SRC/DST pointers.
            std::unordered_map<int, dnnl::memory> args;
            int64_t src_offset = static_cast<int64_t>(batch_token_start) * static_cast<int64_t>(src_row_bytes);
            int64_t dst_offset = static_cast<int64_t>(batch_token_start) * static_cast<int64_t>(dst_row_bytes);

            args.insert({DNNL_ARG_SRC, make_shifted_grouped_memory(batch_pd.src_desc(), input, *_scratch_offsets, src_offset, eng)});
            args.insert({DNNL_ARG_DST, make_shifted_grouped_memory(batch_pd.dst_desc(), output, *_scratch_offsets, dst_offset, eng)});
            args.insert({DNNL_ARG_WEIGHTS, weights.get_onednn_memory(batch_pd.weights_desc(), 0)});

            // Scratchpad (mode=user): must be supplied explicitly for correct results.
            if (pd_prim.scratchpad)
                args.insert({DNNL_ARG_SCRATCHPAD, pd_prim.scratchpad->get_onednn_memory(pd_prim.scratchpad_md, 0)});

            if (moe_cfg.is_weight_quantized) {
                auto& wei_scales = instance.input_memory(moe_cfg.weight_scale_idx);
                auto wei_scales_shape = wei_scales.get_layout().get_shape();
                dnnl::memory::dim d0 = static_cast<dnnl::memory::dim>(num_groups);
                dnnl::memory::dim d1 = wei_scales_shape[1];
                dnnl::memory::dim d2 = wei_scales_shape[2];
                dnnl::memory::dims wei_scales_dims = (moe_cfg.weight_group_size == -1)
                    ? dnnl::memory::dims{d0, d1}
                    : dnnl::memory::dims{d0, d2, d1};
                dnnl::memory::format_tag wei_scales_fmt = (moe_cfg.weight_group_size == -1)
                    ? dnnl::memory::format_tag::ab
                    : dnnl::memory::format_tag::abc;
                dnnl::memory::desc wei_scales_md(
                    wei_scales_dims, convert_data_type(wei_scales.get_layout().data_type), wei_scales_fmt);
                args.insert({DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales.get_onednn_memory(wei_scales_md, 0)});

                if (!moe_cfg.is_weight_symmetric_quantized) {
                    auto& wei_zp = instance.input_memory(moe_cfg.weight_zp_idx);
                    dnnl::memory::desc wei_zp_md(
                        wei_scales_dims, convert_data_type(wei_zp.get_layout().data_type), wei_scales_fmt);
                    args.insert({DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, wei_zp.get_onednn_memory(wei_zp_md, 0)});
                }
            }

            if (_otd_has_bias) {
                auto& bias = instance.input_memory(moe_gemm::MoEGemmInputIdx::BIAS);
                // Bias is a full [num_experts, N] buffer (not OTD-compacted). Gather the
                // batch's experts' bias rows into a scratch buffer so group g maps to the
                // expert loaded into slot g (matching weights/scales/zp compaction).
                const size_t bias_row_bytes =
                    static_cast<size_t>(_otd_N) * dnnl::memory::data_type_size(_otd_bias_dt);
                for (int64_t g = 0; g < num_groups; g++) {
                    const size_t expert = batch_experts[g];
                    _scratch_bias->copy_from(stream, bias,
                                             expert * bias_row_bytes,   // src offset
                                             static_cast<size_t>(g) * bias_row_bytes,  // dst offset
                                             bias_row_bytes, true);
                }
                dnnl::memory::desc bias_md({num_groups, _otd_N}, _otd_bias_dt, dnnl::memory::format_tag::ab);
                args.insert({DNNL_ARG_BIAS, _scratch_bias->get_onednn_memory(bias_md, 0)});
            }

            // Execute this batch
            try {
                batch_prim.execute(stream.get_onednn_stream(), args);
                // Serialize batches: the next batch's load_experts overwrites the
                // weight/scale/zp slots, so the matmul reading them must finish first.
                stream.finish();
            } catch (dnnl::error& err) {
                OPENVINO_THROW("OTD batched execution failed at batch_start=", batch_start,
                    " batch_token_start=", batch_token_start,
                    " M=", M, " num_groups=", num_groups,
                    " error: ", err.what());
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<moe_gemm>& instance) override {
        if (!_otd_ctx || _lru_expert_num == 0) {
            return parent::execute_impl(events, instance);
        }

        // OTD path: load needed experts, remap offsets, execute with reduced weight buffer
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto net_id = network.get_id();

        // Load experts on-demand and build remapped offset buffer
        bool already_executed = prepare_otd_execution(instance);

        event::ptr event;
        if (already_executed) {
            // Batched execution already done in prepare_otd_execution
            if (instance.needs_completion_event())
                event = stream.enqueue_marker({});
        } else {
            // Use OTD arguments with remapped offset buffer
            _args[net_id] = get_arguments_otd(instance);

            // Execute oneDNN primitive
            if (!instance.can_be_optimized()) {
                try {
                    _prim.execute(stream.get_onednn_stream(), _args[net_id]);
                } catch (dnnl::error& err) {
                    OPENVINO_THROW(err.what());
                }
                if (instance.needs_completion_event())
                    event = stream.enqueue_marker({});
            }
        }

        // After gemm_down, invalidate slot mapping for next iteration
        if (!_is_up_gemm) {
            _otd_ctx->slots_valid = false;
        }

        return event;
    }

    static std::shared_ptr<dnnl::matmul::primitive_desc>
        get_moe_gemm_primitive_descriptor(const kernel_impl_params& impl_params,
                                          const dnnl::primitive_attr& attr = dnnl::primitive_attr(),
                                          size_t lru_expert_num_override = 0) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<moe_gemm>();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(impl_params);

        auto input_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::INPUT);
        auto weights_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
        auto output_layout = impl_params.get_output_layout();

        dnnl::memory::dim total_tokens = prim->has_batch_dim ? input_layout.get_shape()[1] : input_layout.get_shape()[0];
        const auto& experts_weight_shape = weights_layout.get_shape();
        dnnl::memory::dim N = experts_weight_shape[1];
        dnnl::memory::dim K = experts_weight_shape.size() == 4 ? experts_weight_shape[2] * experts_weight_shape[3] : experts_weight_shape[2];
        // When OTD is enabled, use lru_expert_num instead of the full num_experts
        dnnl::memory::dim num_experts = lru_expert_num_override > 0
            ? static_cast<dnnl::memory::dim>(lru_expert_num_override)
            : experts_weight_shape[0];

        dnnl::memory::dims input_dims = {total_tokens, K};
        dnnl::memory::dims weights_dims = {num_experts, K, N};
        dnnl::memory::dims output_dims = {total_tokens, N};

        auto input_md = dnnl::memory::desc::grouped(
                input_dims, convert_data_type(input_layout.data_type), 0, num_experts);
        auto output_md = dnnl::memory::desc::grouped(
                output_dims, convert_data_type(output_layout.data_type), 0, num_experts);
        auto weights_md = dnnl::memory::desc(
                weights_dims, convert_data_type(weights_layout.data_type), dnnl::memory::format_tag::acb);

        if (moe_cfg.has_bias) {
            auto bias_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::BIAS);
            auto bias_md = dnnl::memory::desc({num_experts, N}, convert_data_type(bias_layout.data_type), dnnl::memory::format_tag::ab);

            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                input_md,
                weights_md,
                bias_md,
                output_md,
                attr);
        } else {
            return std::make_shared<dnnl::matmul::primitive_desc>(
                engine.get_onednn_engine(),
                input_md,
                weights_md,
                output_md,
                attr);
        }
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    static std::unique_ptr<primitive_impl> create(const moe_gemm_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim = impl_params.typed_desc<moe_gemm>();
        auto moe_cfg = MoEGemmImplementationManager::get_moe_cfg(impl_params);

        if (moe_cfg.is_weight_quantized) {
            if (moe_cfg.weight_group_size == -1) {
                attr->set_scales(DNNL_ARG_WEIGHTS,
                                 (1 << 0) | (1 << 2),
                                 {},
                                 convert_data_type(impl_params.get_input_layout(moe_cfg.weight_scale_idx).data_type));
            } else {
                attr->set_scales(DNNL_ARG_WEIGHTS,
                                 (1 << 0) | (1 << 1) | (1 << 2),
                                 {moe_cfg.weight_group_size, 1},
                                 convert_data_type(impl_params.get_input_layout(moe_cfg.weight_scale_idx).data_type));
            }

            if (!moe_cfg.is_weight_symmetric_quantized) {
                if (moe_cfg.weight_group_size == -1) {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS,
                                         (1 << 0) | (1 << 2),
                                         {},
                                         convert_data_type(impl_params.get_input_layout(moe_cfg.weight_zp_idx).data_type));
                } else {
                    attr->set_zero_points(DNNL_ARG_WEIGHTS,
                                          (1 << 0) | (1 << 1) | (1 << 2),
                                          {moe_cfg.weight_group_size, 1},
                                          convert_data_type(impl_params.get_input_layout(moe_cfg.weight_zp_idx).data_type));
                }
            }
        }

        // For 2GEMM OTD, use lru_expert_num as the grouped matmul group count.
        // Weight buffer is physically sized for lru_expert_num experts (partial upload).
        const size_t lru_expert_num = prim->_otd.lru_expert_num;
        auto prim_desc = get_moe_gemm_primitive_descriptor(impl_params, *attr,
            lru_expert_num > 0 ? lru_expert_num : 0);

        auto impl = std::make_unique<moe_gemm_onednn>(engine, config, attr, *prim_desc);

        // Set up OTD context if enabled
        if (lru_expert_num > 0) {
            auto weights_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::WEIGHT);
            const size_t num_experts = weights_layout.get_shape()[0];
            impl->_lru_expert_num = lru_expert_num;
            impl->_is_up_gemm = prim->is_up_gemm;

            // Store attr and geometry for building per-batch PDs during prefill overflow
            impl->_otd_attr = attr;
            auto input_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::INPUT);
            auto output_layout = impl_params.get_output_layout();
            const auto& experts_weight_shape = weights_layout.get_shape();
            impl->_otd_N = static_cast<dnnl::memory::dim>(experts_weight_shape[1]);
            impl->_otd_K = static_cast<dnnl::memory::dim>(
                experts_weight_shape.size() == 4 ? experts_weight_shape[2] * experts_weight_shape[3]
                                                 : experts_weight_shape[2]);
            impl->_otd_src_dt = convert_data_type(input_layout.data_type);
            impl->_otd_dst_dt = convert_data_type(output_layout.data_type);
            impl->_otd_wei_dt = convert_data_type(weights_layout.data_type);
            impl->_otd_has_bias = moe_cfg.has_bias;
            if (moe_cfg.has_bias) {
                auto bias_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::BIAS);
                impl->_otd_bias_dt = convert_data_type(bias_layout.data_type);
            }

            // Allocate scratch offset buffer [lru_expert_num] i32 USM
            auto offset_layout = cldnn::layout(
                ov::Shape{lru_expert_num}, cldnn::data_types::i32, cldnn::format::bfyx);
            impl->_scratch_offsets = engine.allocate_memory(offset_layout, cldnn::allocation_type::usm_host, false);

            // Allocate scratch bias buffer [lru_expert_num, N] to gather per-batch expert bias.
            // Bias is not OTD-compacted, so each batch must gather its experts' bias rows.
            if (moe_cfg.has_bias) {
                auto bias_layout = impl_params.get_input_layout(moe_gemm::MoEGemmInputIdx::BIAS);
                auto sb_layout = cldnn::layout(
                    ov::Shape{lru_expert_num, static_cast<size_t>(impl->_otd_N)},
                    bias_layout.data_type, cldnn::format::bfyx);
                impl->_scratch_bias = engine.allocate_memory(sb_layout, cldnn::allocation_type::usm_device, false);
            }

            std::string prim_id = prim->id;
            std::string moe_layer_id = prim_id;
            auto pos = prim_id.rfind("_moe_gemm_");
            if (pos != std::string::npos)
                moe_layer_id = prim_id.substr(0, pos);

            impl->_otd_ctx = Moe2GemmOtdRegistry::instance().get_or_create(
                moe_layer_id,
                lru_expert_num,  // capacity = LRU expert count
                num_experts,     // total experts in model
                prim->_moe_config,
                prim->_otd.weight_bin_offsets,
                prim->_otd.weights_path);
        }

        return impl;
    }
};

std::unique_ptr<primitive_impl> MoEGemmImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const  {
    OPENVINO_ASSERT(node.is_type<moe_gemm>());
    return onednn::moe_gemm_onednn::create(static_cast<const moe_gemm_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::moe_gemm)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::moe_gemm_onednn)
