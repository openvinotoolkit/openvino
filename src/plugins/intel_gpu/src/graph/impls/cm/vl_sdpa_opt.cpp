// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "vl_sdpa_opt.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/primitives/vl_sdpa.hpp"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"
#include "registry/implementation_manager.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {

constexpr auto get_vlsdpa_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

constexpr size_t round_up_to_tile(size_t value, size_t tile_size) {
    return (value + tile_size - 1) / tile_size * tile_size;
}

constexpr size_t get_default_kv_blk(size_t head_size) {
    const size_t padded_head_size = round_up_to_tile(head_size, 16);
    const size_t tail_size = padded_head_size - head_size;

    if (padded_head_size <= 64 || tail_size <= 1) {
        return 2;
    }

    return 1;
}

class VLSDPAGenerator : public KernelGenerator {
public:
    VLSDPAGenerator() : KernelGenerator("cm_sdpa_vlen") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_vlsdpa_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);

        // transpose shape into BHLS(4D), or HLS(3D)
        auto transpose_pshape = [](const ov::PartialShape& pshape, const std::vector<int64_t>& order) {
            if (order.empty())
                return pshape;

            auto transposed_pshape = ov::PartialShape::dynamic(pshape.rank());
            for (size_t i = 0; i < order.size(); i++) {
                transposed_pshape[i] = pshape[order[i]];
            }
            return transposed_pshape;
        };

        auto desc = params.typed_desc<vl_sdpa>();
        const auto query_shape = transpose_pshape(params.get_input_layout(0).get_partial_shape(), desc->input_q_transpose_order);
        const auto key_shape = transpose_pshape(params.get_input_layout(1).get_partial_shape(), desc->input_k_transpose_order);

        const size_t head_size = key_shape[query_shape.size() - 1].get_length();
        const size_t num_q_heads = query_shape[query_shape.size() - 3].get_length();
        const size_t num_kv_heads = key_shape[key_shape.size() - 3].get_length();
        const float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_size));
        const size_t cmfla_kv_blk = get_default_kv_blk(head_size);

        GPU_DEBUG_TRACE_DETAIL << "VLSDPA query_shape " << query_shape << ", q_transpose_order " << PartialShape(desc->input_q_transpose_order)
                               << ", key_shape " << key_shape << ", k_transpose_order " << PartialShape(desc->input_k_transpose_order)
                               << ", head_size=" << head_size << ", num_q_heads=" << num_q_heads << ", num_kv_heads=" << num_kv_heads << '\n';

        // Detect whether Q/K/V share a packed buffer (in-place crop from
        // TransposeSplitMatcher axis=1).  Any non-zero padding on an input means the
        // SVM base pointer aliases the packed [L, 3*H, S] buffer, so the kernel must
        // use the packed per-token stride instead of the contiguous one.  Use the
        // layout padding predicate instead of a hard-coded feature-axis index so the
        // check stays correct regardless of tensor rank (3D HLS vs 4D BHLS).
        const bool is_qkv_fused =
            static_cast<bool>(params.input_layouts[0].data_padding) ||
            static_cast<bool>(params.input_layouts[1].data_padding) ||
            static_cast<bool>(params.input_layouts[2].data_padding);

        jit.add({
            make_jit_constant("KERNEL_NAME", get_entry_point(params)),
            make_jit_constant("CMFLA_NUM_HEADS", num_q_heads),
            make_jit_constant("CMFLA_NUM_KV_HEADS", num_kv_heads),
            make_jit_constant("CMFLA_HEAD_SIZE", head_size),
            make_jit_constant("CMFLA_SCALE_FACTOR", scale_factor),
            make_jit_constant("CMFLA_KV_BLK", cmfla_kv_blk),
            make_jit_constant("CMFLA_IS_QKV_FUSED", is_qkv_fused ? 1 : 0),
        });

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;

        for (uint32_t i = 0; i < params.input_layouts.size() - 1; i++) {  // inputs: q, k, v
            args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }

        for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(params.input_layouts.size() - 1)});  // input: cu_seq_lens

        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // need_wg_mapping

        // token_offset_q/k/v: per-slice base offsets applied to the Q/K/V SVM pointers
        // when an in-place crop (TransposeSplitMatcher axis=1 pattern) has propagated a
        // dynamic padding offset into the input layout.  Without these the CM kernel
        // would read from the base of the packed QKV buffer instead of the correct
        // slice.  Each offset equals lower_pad[feature] * head_size (elements); the
        // packed per-token stride is selected separately via CMFLA_IS_QKV_FUSED.
        // When no crop optimization fires the padding is 0 and the offsets are 0.
        args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // token_offset_q
        args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // token_offset_k
        args.push_back({ArgumentDescriptor::Types::SCALAR, 3});  // token_offset_v

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams*) {
            assert(!params.is_dynamic());
            auto desc = params.typed_desc<vl_sdpa>();

            // transpose shape into BHLS(4D), or HLS(3D)
            auto transpose_pshape = [](const ov::Shape& pshape, const std::vector<int64_t>& order) {
                if (order.empty())
                    return pshape;

                auto transposed_pshape = ov::Shape(pshape.size());
                for (size_t i = 0; i < order.size(); i++) {
                    transposed_pshape[i] = pshape[order[i]];
                }
                return transposed_pshape;
            };
            const auto query_shape = transpose_pshape(params.get_input_layout(0).get_shape(), desc->input_q_transpose_order);
            const size_t num_q_heads = query_shape[query_shape.size() - 3];

            // Read cu_seqlens via mem_lock — zero-cost on USM-host (PC), no GPU sync stall.
            const auto cu_seqlens_mem = params.memory_deps.at(params.input_layouts.size() - 1);
            mem_lock<int32_t, mem_lock_type::read> cu_seqlens_lock(cu_seqlens_mem, *params.strm);

            size_t max_seq_len = 0;
            for (size_t i = 1; i < cu_seqlens_lock.size(); i++) {
                auto start_idx = cu_seqlens_lock[i - 1];
                auto end_idx   = cu_seqlens_lock[i];
                max_seq_len = std::max(max_seq_len, static_cast<size_t>(end_idx - start_idx));
            }

            const auto& info = params.get_device_info();
            const size_t CM_GRF_WIDTH = (info.arch <= gpu_arch::xe_hpc) ? 256 : 512;
            const size_t q_step = static_cast<size_t>(std::floor(CM_GRF_WIDTH / 32));
            size_t wg_size = static_cast<size_t>(std::floor((max_seq_len + q_step - 1) / q_step));
            int32_t need_wg_mapping = 0;
            if (wg_size > 16) {
                // seq_len is too large for a single work-group; use wg_size=16 and let
                // the kernel's while-loop scan cu_seqlens to find its sequence/block.
                need_wg_mapping = 1;
                wg_size = 16;
            }

            size_t wg_count;
            if (need_wg_mapping) {
                wg_count = 0;
                const auto wg_seq_len = wg_size * q_step;
                for (size_t i = 1; i < cu_seqlens_lock.size(); i++) {
                    auto start_idx = cu_seqlens_lock[i - 1];
                    auto end_idx   = cu_seqlens_lock[i];
                    wg_count += static_cast<size_t>((end_idx - start_idx + wg_seq_len - 1) / wg_seq_len);
                }
            } else {
                wg_count = cu_seqlens_lock.size() - 1;
            }

            auto& wgs = kd.params.workGroups;
            wgs.global = {num_q_heads, wg_count * wg_size, 1};
            wgs.local = {1, wg_size, 1};

            // Compute element-count offsets arising from in-place crop dynamic padding
            // on Q/K (token_offset_q) and K/V (token_offset_kv) inputs.  These are set
            // when TransposeSplitMatcher replaces Transpose+Split(axis=0) with
            // Split(axis=1): the in-place crop propagates a padding offset into the
            // input layout so Q starts at 0, K at H*S and V at 2*H*S inside the packed
            // buffer.  layout::get_linear_offset() derives this element offset directly
            // from the padded dims/strides, so it is correct for both 3D (HLS) and 4D
            // (BHLS) layouts without a hard-coded feature-axis index or an explicit
            // head_size multiplier.  The packed per-token stride is applied separately
            // inside the kernel via CMFLA_IS_QKV_FUSED.
            const int32_t token_offset_q = static_cast<int32_t>(params.input_layouts[0].get_linear_offset());
            const int32_t token_offset_k = static_cast<int32_t>(params.input_layouts[1].get_linear_offset());
            const int32_t token_offset_v = static_cast<int32_t>(params.input_layouts[2].get_linear_offset());

            std::vector<int32_t> scalars{need_wg_mapping, token_offset_q, token_offset_k, token_offset_v};
            kd.params.scalars.clear();
            for (auto i : scalars) {
                scalar_desc s;
                s.t = scalar_desc::Types::INT32;
                s.v.s32 = static_cast<int32_t>(i);
                kd.params.scalars.push_back(s);
            }
        }};
    }
};

class VLSDPAOptImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::VLSDPAOptImpl)

    Stage::Ptr vl_sdpa = make_stage<VLSDPAGenerator>();

    VLSDPAOptImpl() : PrimitiveImplOCL(VLSDPAOptImplementationManager::get_type_info_static()) {}
    VLSDPAOptImpl(const program_node& node, const RuntimeParams& params) : VLSDPAOptImpl() {
        add_stage(vl_sdpa, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<VLSDPAOptImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> VLSDPAOptImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<vl_sdpa>());
    return std::make_unique<VLSDPAOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::vl_sdpa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::VLSDPAOptImpl)
